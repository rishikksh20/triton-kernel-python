"""FlashAttention forward statistics and explicit first-order backward kernels.

Buffers: contiguous finite FP32 Q/K/V/Out/DO/DQ/DK/DV (B,H,T,D),
LSE/Delta (B,H,T). DO is the upstream gradient; DQ/DK/DV are outputs.
Use 1 <= T <= 4096, 1 <= D <= 128, BH=B*H <= 65535, Scale=D**-0.5.
BD=max(32,triton.next_power_of_2(D)); G=(triton.cdiv(T,32),BH).
Use num_warps=4 and BLOCK_M=BLOCK_N=32, BLOCK_D=BD for the tiled kernels.

Launch in order on the same stream:
1. flash_forward_kernel[G](Q,K,V,Out,LSE,Scale,M=T,N=T,D=D,CAUSAL=CAUSAL,
       BLOCK_M=32,BLOCK_N=32,BLOCK_D=BD,num_warps=4)
2. flash_delta_kernel[(BH*T,)](Out,DO,Delta,D=D,BLOCK_D=BD,num_warps=4)
3. flash_dq_kernel[G](Q,K,V,DO,LSE,Delta,DQ,Scale,T=T,D=D,CAUSAL=CAUSAL,
       BLOCK_M=32,BLOCK_N=32,BLOCK_D=BD,num_warps=4)
4. flash_dkdv_kernel[G](Q,K,V,DO,LSE,Delta,DK,DV,Scale,T=T,D=D,CAUSAL=CAUSAL,
       BLOCK_M=32,BLOCK_N=32,BLOCK_D=BD,num_warps=4)
Skip BH=0. If Out/LSE are saved from an earlier forward, start at stage 2;
they must match Q/K/V, Scale and CAUSAL. Delta is sum_d(Out*DO) for each row.
The caller handles gradient integration. No attention dropout is included.

Source: tutorials/17_flashattention_backward.md

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


@triton.jit
def flash_delta_kernel(Out, DO, Delta, D: tl.constexpr, BLOCK_D: tl.constexpr):
    # Flatten (batch,head,query) to give each program one dot product.
    row = tl.program_id(0)
    features = tl.arange(0, BLOCK_D)
    output = tl.load(Out + row * D + features, mask=features < D, other=0.0)
    upstream = tl.load(DO + row * D + features, mask=features < D, other=0.0)
    # Delta=sum(Out*DO) is also sum_j(P_ij*dP_ij).
    # This identity avoids storing a full probability-gradient matrix.
    tl.store(Delta + row, tl.sum(output * upstream, axis=0))


@triton.jit
def flash_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DQ,
    Scale,
    T: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # This program owns a query tile's dQ; no other program writes it.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, BLOCK_D)
    local_keys = tl.arange(0, BLOCK_N)
    q = tl.load(
        Q + bh * T * D + rows[:, None] * D + features[None, :],
        mask=(rows[:, None] < T) & (features[None, :] < D),
        other=0.0,
    )
    do = tl.load(
        DO + bh * T * D + rows[:, None] * D + features[None, :],
        mask=(rows[:, None] < T) & (features[None, :] < D),
        other=0.0,
    )
    lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=0.0)
    delta = tl.load(Delta + bh * T + rows, mask=rows < T, other=0.0)
    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    end = T
    if CAUSAL:
        end = tl.minimum(T, (query_tile + 1) * BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys = start + local_keys
        kt = tl.load(
            K + bh * T * D + keys[None, :] * D + features[:, None],
            mask=(keys[None, :] < T) & (features[:, None] < D),
            other=0.0,
        )
        vt = tl.load(
            V + bh * T * D + keys[None, :] * D + features[:, None],
            mask=(keys[None, :] < T) & (features[:, None] < D),
            other=0.0,
        )
        scores = tl.dot(q, kt, input_precision="ieee") * Scale
        allowed = (rows[:, None] < T) & (keys[None, :] < T)
        if CAUSAL:
            allowed = allowed & (keys[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        # Reconstruct P=exp(score-LSE) from saved natural-log statistics.
        p = tl.exp(scores - lse[:, None])
        # dP=DO @ V.T; softmax gives dS=P*(dP-Delta).
        dp = tl.dot(do, vt, input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dq = tl.dot(ds, tl.trans(kt), dq, input_precision="ieee")
    # Scores were scaled by Scale, so dQ needs the same factor.
    tl.store(
        DQ + bh * T * D + rows[:, None] * D + features[None, :],
        dq * Scale,
        mask=(rows[:, None] < T) & (features[None, :] < D),
    )


@triton.jit
def flash_dkdv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    Delta,
    DK,
    DV,
    Scale,
    T: tl.constexpr,
    D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # This program owns one key tile's dK and dV across all queries.
    bh = tl.program_id(1)
    keys = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    features = tl.arange(0, BLOCK_D)
    local_rows = tl.arange(0, BLOCK_M)
    k = tl.load(
        K + bh * T * D + keys[:, None] * D + features[None, :],
        mask=(keys[:, None] < T) & (features[None, :] < D),
        other=0.0,
    )
    v = tl.load(
        V + bh * T * D + keys[:, None] * D + features[None, :],
        mask=(keys[:, None] < T) & (features[None, :] < D),
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    for start in range(0, T, BLOCK_M):
        rows = start + local_rows
        q = tl.load(
            Q + bh * T * D + rows[:, None] * D + features[None, :],
            mask=(rows[:, None] < T) & (features[None, :] < D),
            other=0.0,
        )
        do = tl.load(
            DO + bh * T * D + rows[:, None] * D + features[None, :],
            mask=(rows[:, None] < T) & (features[None, :] < D),
            other=0.0,
        )
        lse = tl.load(LSE + bh * T + rows, mask=rows < T, other=0.0)
        delta = tl.load(Delta + bh * T + rows, mask=rows < T, other=0.0)
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * Scale
        allowed = (rows[:, None] < T) & (keys[None, :] < T)
        if CAUSAL:
            allowed = allowed & (keys[None, :] <= rows[:, None])
        scores = tl.where(allowed, scores, -float("inf"))
        # Recompute the probability tile rather than loading saved probabilities.
        p = tl.exp(scores - lse[:, None])
        # dS=P*(DO @ V.T-Delta); accumulate dK=dS.T @ Q and dV=P.T @ DO.
        dp = tl.dot(do, tl.trans(v), input_precision="ieee")
        ds = p * (dp - delta[:, None])
        dk = tl.dot(tl.trans(ds), q, dk, input_precision="ieee")
        dv = tl.dot(tl.trans(p), do, dv, input_precision="ieee")
    # This key tile has a single owner, so gradient stores need no atomics.
    # Apply Scale to dK only; dV is the gradient of the unscaled value product.
    tl.store(
        DK + bh * T * D + keys[:, None] * D + features[None, :],
        dk * Scale,
        mask=(keys[:, None] < T) & (features[None, :] < D),
    )
    tl.store(
        DV + bh * T * D + keys[:, None] * D + features[None, :],
        dv,
        mask=(keys[:, None] < T) & (features[None, :] < D),
    )
