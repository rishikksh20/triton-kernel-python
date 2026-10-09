"""Elementwise ReLU, SiLU, exact GELU, and tanh-approximate GELU.

Buffers: contiguous FP32 X and Y, each with N elements.
KIND: "relu", "silu", "gelu", or "gelu_tanh"; a compile-time choice.
Launch: activation_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
    X, Y, N, KIND="silu", BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
Use BLOCK_SIZE=128, 256, 512, or 1024. Skip N=0.
apply_activation is a device helper, called inside the kernel.

Source: tutorials/02_relu_silu_gelu.md

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
def activation_kernel(X, Y, N, KIND: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # A program owns BLOCK_SIZE consecutive elements, not one CUDA thread.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N
    # The final tile may extend beyond N; masks prevent invalid accesses.
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)
    # Apply the chosen formula locally, then store only real elements.
    result = apply_activation(values, KIND)
    tl.store(Y + offsets, result, mask=valid)
