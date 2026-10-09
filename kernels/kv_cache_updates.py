"""Append one new key/value per batch/head to preallocated KV caches.

Buffers: contiguous FP32 KC/VC (B,H,C,D), NK/NV (B,H,D),
INT32 POS/LENGTHS (B,). LENGTHS is an output, not the old cache length.
Require B,H,C>0, 1<=D<=128, cache element count < 2**31, 0<=POS[b]<C.
Launch: append_kernel[(B*H,)](KC,VC,NK,NV,POS,LENGTHS,H,C,D,
    triton.next_power_of_2(D),num_warps=4)
The cache buffers are modified in place. Only the indicated slot is written.
For true appends, POS[b] equals the existing valid length, with earlier
slots already initialized. Output LENGTHS[b]=POS[b]+1.
Launch cache readers after this entire kernel completes on the same stream;
the length store is not a synchronization signal between head programs.

Source: tutorials/21_kv_cache_updates.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def append_kernel(
    KC,
    VC,
    NK,
    NV,
    POS,
    LENGTHS,
    H: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program owns one (batch,head) cache append.
    row = tl.program_id(0)
    batch, head = row // H, row % H
    feature = tl.arange(0, BLOCK)
    position = tl.load(POS + batch)
    # Cache layout is (B,H,C,D): jump to this head's requested token slot.
    offset = (row * C + position) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, feature < D)
    tl.store(VC + offset, v, feature < D)
    # Exactly one head writes the new valid length, avoiding duplicate stores.
    # Other head programs still finish independently; synchronize by kernel order.
    if head == 0:
        tl.store(LENGTHS + batch, position + 1)
