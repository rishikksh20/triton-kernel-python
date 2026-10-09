"""Seeded inverted dropout; the same kernel also applies its first-order gradient.

Buffers: contiguous FP32 X/Y with N < 2**31 elements; optional BOOL KEEP (N,).
Use P in [0,1], integer SEED in [0,2**31), BLOCK=128,256,512,1024.
Launch: dropout_kernel[(triton.cdiv(N,BLOCK),)](
    X,Y,KEEP,N,P,SEED,SAVE_MASK=True,BLOCK=BLOCK,num_warps=4)
Skip N=0. With SAVE_MASK=False, KEEP can be any valid unused pointer, e.g. Y.
For evaluation use P=0. P=1 produces all zeros without division by zero.
For backward, launch on the contiguous upstream gradient with the same N,
effective P and SEED as forward. This regenerates the original mask; no
saved mask or gradient framework code is needed in the kernel file.

Source: tutorials/23_dropout.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    X,
    Y,
    KEEP,
    N: tl.constexpr,
    P: tl.constexpr,
    SEED,
    SAVE_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Absolute element indices make random masks independent of tile boundaries.
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0)
    # Compile-time endpoint branches avoid singular scaling when P=1.
    if P == 0:
        keep = tl.full((BLOCK,), True, tl.int1)
        y = x
    elif P == 1:
        keep = tl.full((BLOCK,), False, tl.int1)
        y = tl.zeros((BLOCK,), tl.float32)
    else:
        # (SEED,offset) deterministically regenerates the same keep decision.
        keep = tl.rand(SEED, offset) >= P
        # Inverted dropout preserves expectation by scaling kept values by 1/(1-P).
        y = tl.where(keep, x / (1 - P), 0.0)
    tl.store(Y + offset, y, offset < N)
    # Mask storage is optional; backward can regenerate decisions from the seed.
    if SAVE_MASK:
        tl.store(KEEP + offset, keep, offset < N)
