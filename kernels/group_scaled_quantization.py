"""Symmetric INT8 feature-group quantization and group-scaled GEMM.

Buffers: contiguous FP32 X (R,K), INT8 Q (R,K), FP32 S (R,ceil(K/G)).
For GEMM: INT8 A (M,K), BT (N,K), FP32 SA (M,groups), SB (N,groups),
and FP32 C (M,N). BT stores transposed B. All dimensions are positive;
G is 32, 64, or 128 and all buffer element counts are below 2**31.
Quantization uses round-half-away-from-zero and clips to [-127,127].
Zero groups use scale 1. Inputs must be finite with representable scales.

Launches, num_warps=4:
  group_quantize_kernel[(R,triton.cdiv(K,G))](X,Q,S,K,G)
  group_dequantize_kernel[(triton.cdiv(R*K,256),)](Q,S,XHAT,R*K,K,G,256)
  group_scaled_gemm_kernel[(triton.cdiv(M,32),triton.cdiv(N,32))](
      A,BT,SA,SB,C,M,N,K,G,32,32)
Each INT32 group dot is converted/scaled BEFORE adding to the FP32 sum.
Provide matching scales; C approximates X @ W.T, not an unscaled INT8 dot.

Source: tutorials/27_group_scaled_quantization.md
The caller allocates/validates buffers on one CUDA device. Offsets count
elements, not bytes. These files contain only Triton kernels/device code.
"""

import triton
import triton.language as tl


@triton.jit
def group_quantize_kernel(X, Q, S, K: tl.constexpr, G: tl.constexpr):
    # One program owns one row's feature group, including a possible K tail.
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row * K + k, k < K, other=0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
    # Explicit rounding keeps the reference independent of cast rounding modes.
    scaled = x / scale
    magnitude = tl.floor(tl.abs(scaled) + 0.5)
    rounded = tl.where(scaled < 0, -magnitude, magnitude)
    encoded = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
    tl.store(Q + row * K + k, encoded, k < K)
    tl.store(S + row * tl.cdiv(K, G) + group, scale)


@triton.jit
def group_dequantize_kernel(
    Q,
    S,
    XHAT,
    SIZE: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # A feature's scale depends on both its row and its group along K.
    row, feature = index // K, index % K
    q = tl.load(Q + index, index < SIZE, other=0).to(tl.float32)
    scale = tl.load(S + row * tl.cdiv(K, G) + feature // G, index < SIZE, other=0)
    tl.store(XHAT + index, q * scale, index < SIZE)


@triton.jit
def group_scaled_gemm_kernel(
    A,
    BT,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, G)
    groups: tl.constexpr = tl.cdiv(K, G)
    result = tl.zeros((BM, BN), tl.float32)
    for group in range(groups):
        k = group * G + kk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            BT + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        # Reset the integer accumulator for every group: scales change with K.
        partial = tl.dot(a, b, tl.zeros((BM, BN), tl.int32))
        sa = tl.load(SA + rows * groups + group, rows < M, other=0)
        sb = tl.load(SB + cols * groups + group, cols < N, other=0)
        result += partial.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
