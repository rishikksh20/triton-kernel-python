"""Top-2 MoE routing, capacity-limited dispatch, and weighted expert gather.

Buffers: contiguous FP32 Z (T,E), X/Y (T,D), GATES (T,2),
PACKED/EXPERT_OUT (E,CAPACITY,D); INT32 IDS/SLOTS (T,2), COUNTS (E,),
TOKENS (E,CAPACITY). Require finite Z, T>=1, E=2..256, D=1..1024,
CAPACITY>=1 and element counts/counter ranges < 2**31.
All stages use num_warps=4; inputs to dispatch must have two distinct,
in-range expert IDs per token, as produced by route_kernel.

Launch in order on the same stream:
1. route_kernel[(T,)](Z,IDS,GATES,E,triton.next_power_of_2(E),num_warps=4).
2. Initialize COUNTS=0 before every dispatch; initialize unused PACKED=0
   and TOKENS=-1 if later code will read full expert buffers.
3. dispatch_kernel[(T*2,)](X,IDS,COUNTS,PACKED,TOKENS,SLOTS,
       CAPACITY,D,triton.next_power_of_2(D),num_warps=4).
4. Compute expert outputs for populated slots, then
   combine_kernel[(T,)](EXPERT_OUT,IDS,GATES,SLOTS,Y,
       CAPACITY,D,triton.next_power_of_2(D),num_warps=4).
Expert computation is supplied separately. Atomic slot order is unspecified;
COUNTS includes dropped routes, so process only min(COUNTS[e],CAPACITY) slots.
SLOTS >= CAPACITY marks dropped routes; combine omits them without gate
renormalization. CAPACITY=T guarantees no overflow for distinct top-2 IDs.

Source: tutorials/26_moe_routing.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def route_kernel(Z, IDS, GATES, E: tl.constexpr, BLOCK: tl.constexpr):
    # One program chooses two distinct experts for one token.
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    score = tl.load(Z + token * E + expert, expert < E, other=-float("inf"))
    first_score = tl.max(score, axis=0)
    # A minimum-ID reduction breaks equal-logit ties deterministically.
    first = tl.min(
        tl.where((expert < E) & (score == first_score), expert, 2147483647), axis=0
    )
    # Remove the first expert before selecting the second; padded experts cannot win.
    remaining = tl.where(expert != first, score, -float("inf"))
    second_score = tl.max(remaining, axis=0)
    second = tl.min(
        tl.where(
            (expert < E) & (expert != first) & (remaining == second_score),
            expert,
            2147483647,
        ),
        axis=0,
    )
    # A stable two-value softmax normalizes only the chosen expert gates.
    ratio = tl.exp(second_score - first_score)
    first_gate = 1 / (1 + ratio)
    second_gate = ratio / (1 + ratio)
    tl.store(IDS + token * 2, first)
    tl.store(IDS + token * 2 + 1, second)
    tl.store(GATES + token * 2, first_gate)
    tl.store(GATES + token * 2 + 1, second_gate)


@triton.jit
def dispatch_kernel(
    X,
    IDS,
    COUNTS,
    PACKED,
    TOKENS,
    SLOTS,
    CAPACITY: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # There are two route programs per token; each copies one feature vector.
    route = tl.program_id(0)
    token = route // 2
    expert = tl.load(IDS + route)
    # Atomic fetch-add reserves a unique slot within the selected expert.
    # COUNTS must be zeroed before launch; ordering between routes is unspecified.
    slot = tl.atomic_add(COUNTS + expert, 1)
    tl.store(SLOTS + route, slot)
    feature = tl.arange(0, BLOCK)
    values = tl.load(X + token * D + feature, feature < D, other=0)
    # Counters include overflow, but only slots below CAPACITY may write storage.
    accepted = slot < CAPACITY
    tl.store(
        PACKED + (expert * CAPACITY + slot) * D + feature,
        values,
        accepted & (feature < D),
    )
    tl.store(TOKENS + expert * CAPACITY + slot, token, accepted)


@triton.jit
def combine_kernel(
    EXPERT_OUT,
    IDS,
    GATES,
    SLOTS,
    Y,
    CAPACITY: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One owner gathers both routed expert outputs into a single token vector.
    token = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.float32)
    # Two choices are known at compile time, so the loop is unrolled.
    for choice in tl.static_range(2):
        route = token * 2 + choice
        expert = tl.load(IDS + route)
        slot = tl.load(SLOTS + route)
        gate = tl.load(GATES + route)
        # Dropped routes read zero; gates keep their original top-2 normalization.
        values = tl.load(
            EXPERT_OUT + (expert * CAPACITY + slot) * D + feature,
            (slot < CAPACITY) & (feature < D),
            other=0,
        )
        result += gate * values
    tl.store(Y + token * D + feature, result, feature < D)
