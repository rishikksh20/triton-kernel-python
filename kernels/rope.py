"""RoPE with adjacent feature pairs and explicit absolute token positions.

Buffers: contiguous FP32 X/Y (B,H,T,D), COS/SIN (P,D/2), INT32 POS (B,T).
Require nonempty dimensions, even D in [2,128], X element count < 2**31.
All POS entries must be in [0,P). Tables are supplied by the caller.
Launch: rope_kernel[(B*H*T,)](X,POS,COS,SIN,Y,H,T,D,
    triton.next_power_of_2(D//2),num_warps=4,enable_fp_fusion=False)
Each head of a token uses the same absolute position. The layout is
adjacent (even,odd) pairs, not split-half rotation. Disabling FP fusion
matches the tutorial's arithmetic order when comparing outputs.

Source: tutorials/20_rope.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    X,
    POS,
    COS,
    SIN,
    Y,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flattened rows enumerate (batch,head,token), with D consecutive features.
    row = tl.program_id(0)
    # Heads share token positions; dividing by H*T recovers the batch index.
    token = row % T
    batch = row // (H * T)
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    position = tl.load(POS + batch * T + token)
    # Select precomputed trigonometric values at the absolute token position.
    table_offset = position * (D // 2) + pair
    c = tl.load(COS + table_offset, valid, other=0)
    s = tl.load(SIN + table_offset, valid, other=0)
    # Load adjacent x_even/x_odd and rotate using [c,-s; s,c].
    even = tl.load(X + row * D + 2 * pair, valid, other=0)
    odd = tl.load(X + row * D + 2 * pair + 1, valid, other=0)
    tl.store(Y + row * D + 2 * pair, even * c - odd * s, valid)
    tl.store(Y + row * D + 2 * pair + 1, even * s + odd * c, valid)
