"""Fused SwiGLU projections: Y = SiLU(X @ WG) * (X @ WU).

Buffers: contiguous FP32 X (M,K), WG/WU (K,N), Y (M,N).
Use positive M,N,K and matrix element counts < 2**31.
Launch: swiglu_kernel[(triton.cdiv(M,32),triton.cdiv(N,32))](
    X,WG,WU,Y,M,N,K,32,32,32,num_warps=4)
Each program reuses an X tile for two FP32 accumulators, then applies the
gated epilogue. There is no bias, output projection, or backward in this topic.

Source: tutorials/24_swiglu.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def swiglu_kernel(
    X,
    WG,
    WU,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One program owns an output tile and the full K reduction for that tile.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    # Keep two separate FP32 accumulators for the gate and up projections.
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(0, tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        # Reuse the same loaded X tile with both sets of projection weights.
        offsets = k[:, None] * N + cols[None, :]
        mask = (k[:, None] < K) & (cols[None, :] < N)
        g = tl.load(WG + offsets, mask, other=0)
        u = tl.load(WU + offsets, mask, other=0)
        gate = tl.dot(a, g, gate, input_precision="ieee")
        up = tl.dot(a, u, up, input_precision="ieee")
    # Complete both projections before applying SiLU(gate)*up.
    # Intermediate projection matrices are never written to global memory.
    result = (gate / (1 + tl.exp(-gate))) * up
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
