"""Stable row softmax: Y[r,c] = exp(X[r,c]-max_r) / sum_c exp(X[r,c]-max_r).

Buffers: contiguous finite FP32 X/Y (R,N_COLS), 1 <= N_COLS <= 4096.
BLOCK_SIZE=triton.next_power_of_2(N_COLS) covers an entire row.
Launch: softmax_kernel[(R,)](X, Y, N_COLS=N_COLS,
    BLOCK_SIZE=BLOCK_SIZE, num_warps=4 if BLOCK_SIZE <= 1024 else 8)
Skip R=0. Each program independently reduces and normalizes one row.
Padding is -infinity so it contributes neither a maximum nor a numerator.

Source: tutorials/06_softmax.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS

    # Contiguous rows start at row * N_COLS.
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=-float("inf"))
    # Subtracting the row maximum avoids exponential overflow.
    # The padded -infinity lanes exponentiate to zero.
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    # axis=0 reduces the feature vector to one scalar for this row.
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(Y + row * N_COLS + columns, result, mask=valid)
