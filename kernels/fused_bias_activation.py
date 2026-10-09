"""Fused column bias and activation: Y[r,c] = activation(X[r,c] + Bias[c]).

Buffers: contiguous FP32 X/Y (R,N_COLS), Bias (N_COLS,).
TOTAL=R*N_COLS; N_COLS must be positive.
KIND: "relu", "silu", "gelu", or "gelu_tanh".
Launch: bias_activation_kernel[(triton.cdiv(TOTAL, BLOCK_SIZE),)](
    X, Bias, Y, TOTAL, N_COLS=N_COLS, KIND="relu",
    BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
Use BLOCK_SIZE=128, 256, 512, or 1024. Skip zero rows.
The bias-plus-input intermediate stays within the program.

Source: tutorials/03_fused_bias_activation.md

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
def bias_activation_kernel(
    X,
    Bias,
    Y,
    TOTAL,
    N_COLS: tl.constexpr,
    KIND: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Assign a flat output tile to this program.
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < TOTAL

    # 2. Map each matrix element to its column's bias.
    columns = offsets % N_COLS
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(Bias + columns, mask=valid, other=0.0).to(tl.float32)

    # 3. Keep the intermediate local; do not store a full Z tensor.
    biased = values + biases
    activated = apply_activation(biased, KIND)

    # 4. Store the final output only.
    tl.store(Y + offsets, activated, mask=valid)
