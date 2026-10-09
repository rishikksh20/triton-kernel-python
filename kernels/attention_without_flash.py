"""Materialized attention in three stages: Scores -> Probabilities -> Output.

Buffers: contiguous FP32 Q (B,H,M,D), Key (B,H,N,D), V (B,H,N,DV),
Scores/Probabilities (B,H,M,N), Output (B,H,M,DV).
Use D,DV > 0, 1 <= N <= 4096, BH=B*H <= 65535, Scale=D**-0.5.
CAUSAL=True requires square self-attention M=N in this tutorial.

Launch in order on the same stream, with num_warps=4 unless noted:
1. attention_scores_kernel[(triton.cdiv(M,32), triton.cdiv(N,32), BH)](
       Q, Key, Scores, M, N, D, Scale, CAUSAL=CAUSAL,
       BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
2. attention_softmax_kernel[(BH*M,)](Scores, Probabilities,
       N_COLS=N, BLOCK_SIZE=triton.next_power_of_2(N))
   Use 8 warps instead of 4 if the padded row width exceeds 1024.
3. attention_value_kernel[(triton.cdiv(M,32), triton.cdiv(DV,32), BH)](
       Probabilities, V, Output, M, DV, N,
       BLOCK_M=32, BLOCK_N=32, BLOCK_K=32)
Skip BH=0 or M=0. Stage 3's N argument is DV and its K argument is N.
Scores and probabilities occupy quadratic storage; FlashAttention avoids it.

Source: tutorials/12_attention_without_flash.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def attention_scores_kernel(
    Q,
    Key,
    Scores,
    M,
    N,
    D,
    Scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Flatten (batch, head) into one independent matrix pair.
    bh = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    local_d = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for d_tile in range(tl.cdiv(D, BLOCK_K)):
        d = d_tile * BLOCK_K + local_d
        q_ptrs = Q + bh * M * D + rows[:, None] * D + d[None, :]
        # Key is stored as (N,D); this pointer tile presents its transpose.
        k_ptrs = Key + bh * N * D + columns[None, :] * D + d[:, None]
        q = tl.load(q_ptrs, mask=(rows[:, None] < M) & (d[None, :] < D), other=0.0)
        k = tl.load(k_ptrs, mask=(columns[None, :] < N) & (d[:, None] < D), other=0.0)
        acc = tl.dot(q, k, acc, input_precision="ieee")
    # Q @ Key.T yields raw scores; Scale=1/sqrt(D) controls their magnitude.
    scores = acc * Scale
    # Future key columns are -infinity, so their softmax probability is zero.
    if CAUSAL:
        scores = tl.where(columns[None, :] <= rows[:, None], scores, -float("inf"))
    tl.store(
        Scores + bh * M * N + rows[:, None] * N + columns[None, :],
        scores,
        mask=(rows[:, None] < M) & (columns[None, :] < N),
    )


@triton.jit
def attention_softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Flattened row index spans all (batch,head,query) combinations.
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS

    # Contiguous rows start at row * N_COLS.
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=-float("inf"))
    # Stable softmax: subtract the maximum before exponentiation.
    # Causal and padded -infinity scores contribute zero probability.
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(Y + row * N_COLS + columns, result, mask=valid)


@triton.jit
def attention_value_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    # Here batch means one flattened (batch,head) pair.
    # A is probabilities (queries,keys); B is V (keys,value_features).
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + batch * M * K + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + batch * K * N + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    c_ptrs = C + batch * M * N + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)
