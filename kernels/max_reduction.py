"""One stage of a hierarchical FP32 maximum reduction.

Buffers: contiguous FP32 X (N,), Partials (ceil(N/BLOCK_SIZE),); N > 0.
Launch: max_stage_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
    X, Partials, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
Use BLOCK_SIZE=128, 256, 512, or 1024.
Repeat on partial maxima using distinct buffers until one value remains.
Negative infinity is the padding identity, including for all-negative data.
The tutorial assumes finite inputs; an empty vector has no maximum.

Source: tutorials/05_max_reduction.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def max_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # Each program owns one input tile and one partial-output slot.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # Negative infinity cannot win against a valid finite input.
    values = tl.load(X + offsets, mask=valid, other=-float("inf"))
    partial = tl.max(values, axis=0)
    tl.store(Partials + pid, partial)
