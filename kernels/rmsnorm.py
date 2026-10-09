"""RMSNorm: Y[r,c] = X[r,c] * rsqrt(mean_c(X[r,c]^2) + EPS) * Weight[c].

Buffers: contiguous FP32 X/Y (R,N_COLS), Weight (N_COLS,).
Use 1 <= N_COLS <= 4096 and finite EPS > 0.
BLOCK_SIZE=triton.next_power_of_2(N_COLS).
Launch: rmsnorm_kernel[(R,)](X, Weight, Y, EPS,
    N_COLS=N_COLS, BLOCK_SIZE=BLOCK_SIZE,
    num_warps=4 if BLOCK_SIZE <= 1024 else 8)
Skip R=0. Reductions and normalization use FP32 working values.

Source: tutorials/07_rmsnorm.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(X, Weight, Y, EPS, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # One program owns one row; column offsets are shared by all row loads.
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=0.0).to(tl.float32)

    # Padding contributes zero, but the divisor is the real feature count.
    mean_square = tl.sum(values * values, axis=0) / N_COLS
    # EPS prevents division by zero for a row whose mean square is zero.
    inverse_rms = tl.rsqrt(mean_square + EPS)
    # Broadcast the learned feature scale across rows, with no mean subtraction.
    weights = tl.load(Weight + columns, mask=valid, other=0.0).to(tl.float32)
    result = (values * inverse_rms) * weights
    tl.store(Y + row * N_COLS + columns, result, mask=valid)
