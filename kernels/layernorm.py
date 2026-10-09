"""LayerNorm: normalize each row, then apply per-feature weight and bias.

Buffers: contiguous FP32 X/Y (R,N_COLS), Weight/Bias (N_COLS,).
Use 1 <= N_COLS <= 4096 and finite EPS > 0.
BLOCK_SIZE=triton.next_power_of_2(N_COLS).
Launch: layernorm_kernel[(R,)](X, Weight, Bias, Y, EPS,
    N_COLS=N_COLS, BLOCK_SIZE=BLOCK_SIZE,
    num_warps=4 if BLOCK_SIZE <= 1024 else 8)
Skip R=0. Variance divides by N_COLS, not N_COLS-1 or the padded width.

Source: tutorials/08_layernorm.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    X,
    Weight,
    Bias,
    Y,
    EPS,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One program computes the statistics and output of one complete row.
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=0.0).to(tl.float32)
    # Reduce FP32 values and divide by the real, unpadded feature count.
    mean = tl.sum(values, axis=0) / N_COLS

    # Padding must stay zero after centering, not become -mean.
    centered = tl.where(valid, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N_COLS
    normalized = centered * tl.rsqrt(variance + EPS)

    # The learned affine parameters are indexed by feature, not by row.
    weights = tl.load(Weight + columns, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(Bias + columns, mask=valid, other=0.0).to(tl.float32)
    result = normalized * weights + biases
    tl.store(Y + row * N_COLS + columns, result, mask=valid)
