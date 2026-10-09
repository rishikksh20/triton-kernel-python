"""Causal FlashAttention: query i may attend only to keys j <= i.

Buffers: contiguous finite FP32 Q/K/V/Out (T,D), LSE (T,).
Use 1 <= T <= 4096, 1 <= D <= 128, Scale=D**-0.5.
BD=max(32,triton.next_power_of_2(D)).
Launch: flash_forward_kernel[(triton.cdiv(T,32), 1)](
    Q, K, V, Out, LSE, Scale, M=T, N=T, D=D, CAUSAL=True,
    BLOCK_M=32, BLOCK_N=32, BLOCK_D=BD, num_warps=4)
Square self-attention uses matching query/key position origins.
Future tiles are skipped; a per-element mask handles the diagonal tile.
LSE is a required FP32 output containing natural-log normalization statistics.

Source: tutorials/15_causal_flashattention.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def flash_forward_kernel(
    Q,
    K,
    V,
    Out,
    LSE,
    Scale,
    M: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program owns a query tile within one batch/head pair.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, BLOCK_D)
    local_keys = tl.arange(0, BLOCK_N)
    # Load Q once for this tile. [:,None] and [None,:] broadcast
    # row/feature indices into a two-dimensional pointer tile.
    q = tl.load(
        Q + bh * M * D + rows[:, None] * D + features[None, :],
        mask=(rows[:, None] < M) & (features[None, :] < D),
        other=0.0,
    )
    # Per query row: maximum m, denominator l, and vector numerator.
    # Their sizes depend on tile dimensions, not on the whole key sequence.
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    numerator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    end = N
    # Future key tiles cannot contribute to any query in this tile.
    if CAUSAL:
        end = tl.minimum(N, (query_tile + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys = start + local_keys
        k_transposed = tl.load(
            K + bh * N * D + keys[None, :] * D + features[:, None],
            mask=(keys[None, :] < N) & (features[:, None] < D),
            other=0.0,
        )
        v = tl.load(
            V + bh * N * D + keys[:, None] * D + features[None, :],
            mask=(keys[:, None] < N) & (features[None, :] < D),
            other=0.0,
        )
        # Compute a temporary query-by-key tile; it is never stored to HBM.
        scores = tl.dot(q, k_transposed, input_precision="ieee") * Scale
        allowed = (rows[:, None] < M) & (keys[None, :] < N)
        if CAUSAL:
            allowed = allowed & (keys[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        # Padded query rows must not evaluate -inf - (-inf).
        new_max = tl.where(rows < M, new_max, 0.0)
        correction = tl.exp(running_max - new_max)
        # Keep weights unnormalized until all key tiles have been processed.
        weights = tl.exp(scores - new_max[:, None])
        running_sum = correction * running_sum + tl.sum(weights, axis=1)
        # Rescale the vector numerator as well as the scalar denominator.
        numerator = numerator * correction[:, None]
        numerator = tl.dot(weights, v, numerator, input_precision="ieee")
        running_max = new_max
    safe_sum = tl.where(rows < M, running_sum, 1.0)
    output = numerator / safe_sum[:, None]
    tl.store(
        Out + bh * M * D + rows[:, None] * D + features[None, :],
        output,
        mask=(rows[:, None] < M) & (features[None, :] < D),
    )
    # Natural-log LSE is enough to reconstruct probabilities in backward.
    tl.store(LSE + bh * M + rows, running_max + tl.log(safe_sum), mask=rows < M)
