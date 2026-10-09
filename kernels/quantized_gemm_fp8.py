"""Scaled low-precision GEMM: C = (A_q @ BT_q.T) * SA[:,None] * SB[None,:].

Buffers: contiguous A (M,K), BT (N,K), FP32 SA (M,), SB (N,), C (M,N).
A and BT share INT8 dtype with IS_FP8=False or E4M3 FP8 with IS_FP8=True.
BT is already stored as transposed B; it is not a row-major (K,N) matrix.
Use positive M,N,K, K<=65536, and matrix element counts < 2**31.
Native FP8 requires an NVIDIA GPU with compute capability >= 9.0 and
compatible Triton support. Provide prequantized inputs and matching scales.
Launch: quantized_gemm_kernel[(triton.cdiv(M,32),triton.cdiv(N,32))](
    A,BT,SA,SB,C,M,N,K,IS_FP8,32,32,64,num_warps=4)
INT8 accumulates in INT32; FP8 accumulates in FP32. Scale only after the
complete dot product because these scales are per row, not per K group.

Source: tutorials/19_quantized_gemm_fp8.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def quantized_gemm_kernel(
    A,
    BT,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    IS_FP8: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    # One program owns one BM-by-BN output tile.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    # Compile-time format selection fixes both the dot and accumulator dtype.
    if IS_FP8:
        acc = tl.zeros((BM, BN), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.int32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + kk
        left = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        # BT is stored (N,K); swapping its indexing presents a (BK,BN) tile.
        right = tl.load(
            BT + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        if IS_FP8:
            acc = tl.dot(left, right, acc, max_num_imprecise_acc=0)
        else:
            acc = tl.dot(left, right, acc)
    # Scale vectors broadcast over output rows and columns.
    # This is valid for per-row scales that remain constant across K.
    scale_a = tl.load(SA + rows, rows < M, other=0)
    scale_b = tl.load(SB + cols, cols < N, other=0)
    result = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
