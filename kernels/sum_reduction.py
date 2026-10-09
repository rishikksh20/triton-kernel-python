"""One stage of a hierarchical FP32 sum reduction.

Buffers: contiguous FP32 X (N,), Partials (ceil(N/BLOCK_SIZE),).
Launch: sum_stage_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
    X, Partials, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
Use BLOCK_SIZE=128, 256, 512, or 1024.
To obtain a scalar, launch again on Partials with a new, distinct output
buffer; repeat until one partial remains. A single stage is not a full
reduction when N > BLOCK_SIZE. Never overwrite unread input with partials.
For N=0, launch one program with a valid output slot to write the sum 0.

Source: tutorials/04_sum_reduction.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def sum_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # 1. Each program owns one input tile and one output partial sum.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 2. Zero is the identity for addition, so masked lanes contribute zero.
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)

    # 3. Reduce this program's tile to one FP32 value.
    partial = tl.sum(values, axis=0)

    # 4. Each program writes a different partial-sum slot.
    tl.store(Partials + pid, partial)
