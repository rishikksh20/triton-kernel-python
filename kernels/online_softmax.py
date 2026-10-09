"""Two-pass streaming softmax with a bounded tile and online statistics.

Buffers: contiguous finite FP32 X/Y (R,N_COLS), Maxima/Denominators (R,).
Use 1 <= N_COLS <= 65536 and BLOCK_SIZE=128, 256, 512, or 1024.
Launch: online_softmax_kernel[(R,)](X, Y, Maxima, Denominators,
    N_COLS=N_COLS, BLOCK_SIZE=256, num_warps=4)
Skip R=0. Maxima stores each final row maximum; Denominators stores
sum(exp(x-maximum)), not sum(exp(x)). The second pass writes probabilities.

Source: tutorials/13_online_softmax.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    X,
    Y,
    Maxima,
    Denominators,
    N_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    local_columns = tl.arange(0, BLOCK_SIZE)
    # Maintain m=max(processed scores), l=sum(exp(processed scores-m)).
    running_max = tl.full((), -float("inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)

    # Pass 1: update the maximum and shifted denominator together.
    for start in range(0, N_COLS, BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(
            X + row * N_COLS + columns, mask=columns < N_COLS, other=-float("inf")
        )
        block_max = tl.max(values, axis=0)
        new_max = tl.maximum(running_max, block_max)
        # A larger maximum changes the exponential reference point.
        # Rescale the old denominator before adding the current tile.
        correction = tl.exp(running_max - new_max)
        block_sum = tl.sum(tl.exp(values - new_max), axis=0)
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    # These row-sized outputs describe the completed normalizer.
    tl.store(Maxima + row, running_max)
    tl.store(Denominators + row, running_sum)

    # Pass 2: final normalization needs the completed row statistics.
    for start in range(0, N_COLS, BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(
            X + row * N_COLS + columns, mask=columns < N_COLS, other=-float("inf")
        )
        probabilities = tl.exp(values - running_max) / running_sum
        tl.store(Y + row * N_COLS + columns, probabilities, mask=columns < N_COLS)
