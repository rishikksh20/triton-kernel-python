"""Tiled GEMM with a fused epilogue: Y = activation(A @ B + Bias).

Buffers: contiguous FP32 A (M,K), B (K,N), Bias (N,), Y (M,N).
KIND: "relu", "silu", "gelu", or "gelu_tanh".
Launch: matmul_bias_activation_kernel[(triton.cdiv(M,32), triton.cdiv(N,32))](
    A, B, Bias, Y, M, N, K, KIND="relu",
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4)
Skip M=0 or N=0. K=0 produces activation(Bias), broadcast over rows.
The nonlinear activation runs after the complete K reduction.

Source: tutorials/10_matmul_bias_activation.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def apply_activation(value, KIND: tl.constexpr):
    # KIND is constexpr: the compiler keeps only the selected formula.
    # ReLU clips negative values; SiLU multiplies x by sigmoid(x).
    if KIND == "relu":
        result = tl.maximum(value, 0.0)
    elif KIND == "silu":
        result = value / (1.0 + tl.exp(-value))
    elif KIND == "gelu":
        # Exact GELU uses the standard-normal CDF expressed through erf.
        result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    else:  # KIND must be "gelu_tanh" in this branch.
        # The tanh approximation uses sqrt(2/pi) and a cubic correction.
        inner = 0.7978845608028654 * (value + 0.044715 * value * value * value)
        result = 0.5 * value * (1.0 + libdevice.tanh(inner))
    return result


@triton.jit
def matmul_bias_activation_kernel(
    A,
    B,
    Bias,
    Y,
    M,
    N,
    K,
    KIND: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    # Epilogue: add each column's bias and activate before the only store.
    bias_tile = tl.load(Bias + columns, mask=columns < N, other=0.0)
    result = apply_activation(accumulator + bias_tile[None, :], KIND)
    c_ptrs = Y + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, result, mask=c_valid)
