"""Vector addition: C[i] = A[i] + B[i].

Buffers: contiguous FP32 A, B, C, each with N elements.
Launch: vector_add_kernel[(triton.cdiv(N, BLOCK_SIZE),)](
    A, B, C, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
Use BLOCK_SIZE=128, 256, 512, or 1024. Skip the launch when N=0.
Each program owns a disjoint output tile; masks cover the final partial tile.

Source: tutorials/01_vector_addition.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    # 1. Find the tile owned by this program.
    pid = tl.program_id(axis=0)

    # 2. Compute the logical element indices in that tile.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 3. Load both inputs. Invalid positions receive zero locally.
    a_tile = tl.load(A + offsets, mask=valid, other=0.0)
    b_tile = tl.load(B + offsets, mask=valid, other=0.0)

    # 4. Add the tiles element by element.
    c_tile = a_tile + b_tile

    # 5. Write only real output elements.
    tl.store(C + offsets, c_tile, mask=valid)
