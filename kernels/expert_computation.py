"""Per-expert GEMMs and rank-owned top-2 communication packing/combining.

expert_gemm_kernel: FP32 X (E,CAPACITY,K), W (E,K,N), OUT (E,CAPACITY,N);
INT32 COUNTS (E,), nonnegative requested route counts. Process
min(COUNTS[e],CAPACITY) rows and write zero to every unused output row.
Launch grid (cdiv(CAPACITY,32),cdiv(N,32),E), constexpr CAPACITY,K,N,
BM=BN=BK=32, APPLY_SILU=False/True, num_warps=4. IEEE FP32 dot products.
Chain a SiLU up GEMM and linear down GEMM for a simple two-layer expert.

Rank packing: X (T,D), INT32 IDS/SLOTS (T,2), COUNTS (WORLD,),
SEND_EXPERT/SEND_ROUTE (WORLD,RCAP), FP32 SEND_X (WORLD,RCAP,D).
Global experts are contiguous per owner: owner=id//LOCAL_E, local=id%LOCAL_E.
IDs in [0,WORLD*LOCAL_E), distinct per token. COUNTS=0 EACH dispatch;
unused SEND_EXPERT/SEND_ROUTE=-1, SEND_X=0 for fixed-size communication.
Launch pack_rank_routes_kernel[(2*T,)](...,LOCAL_E,RCAP,D,next_power_of_2(D)).
COUNTS includes overflow; SLOTS>=RCAP is dropped, without renormalization.

Returned data: FP32 RETURNED (WORLD,RCAP,N), GATES (T,2), Y (T,N).
RETURNED must preserve each original destination rank/slot. Receiver expert
dispatch may reorder rows, so restore that order BEFORE the reverse collective.
Launch combine_rank_returns_kernel[(T,)](...,LOCAL_E,RCAP,N,next_power_of_2(N)).

Receiver dispatch: RECV_X (WORLD,RCAP,D), RECV_EXPERT (WORLD,RCAP),
INT32 LOCAL_COUNTS (LOCAL_E,), ROW_SLOTS (WORLD,RCAP), and FP32
PACKED (LOCAL_E,ECAP,D). Valid RECV_EXPERT is in [0,LOCAL_E); -1 is padding.
Zero LOCAL_COUNTS before dispatch_received_kernel[(WORLD*RCAP,)](...,
    ECAP,D,next_power_of_2(D)). Each valid received row reserves one expert slot.
After expert GEMMs, restore_received_kernel[(WORLD*RCAP,)](...,
    ECAP,N,next_power_of_2(N)) writes RETURN_SEND (WORLD,RCAP,N) in original
receive-row order, zero for padded/overflow rows. ECAP=WORLD*RCAP prevents
receiver overflow even if every row selects the same local expert.
Positive dimensions; D,N<=1024 for rank vector kernels; finite inputs;
all element counts and counters <2**31; all buffers on one CUDA device.
Communication/collectives are supplied by the host, outside these kernels.

Source: tutorials/31_expert_computation.md
Offsets count elements, not bytes; raw kernels do not allocate/validate buffers.

Optional balance loss: FP32 LOGITS/DLOGITS (T,E), ROW_LOSS (T,), INT32
COUNTS (E,) from top-2 routing BEFORE dropping routes, sum(COUNTS)=2*T.
Launch load_balance_kernel[(T,)](...,T,E,next_power_of_2(E)), num_warps=4.
sum(ROW_LOSS) = E*sum_e [COUNTS[e]/(2*T)]*mean_t softmax(LOGITS)[t,e].
DLOGITS differentiates this scalar with COUNTS treated as fixed metadata.
It is an auxiliary router objective, not the gradient of the full expert layer.
"""

import triton
import triton.language as tl


@triton.jit
def load_balance_kernel(
    LOGITS,
    COUNTS,
    ROW_LOSS,
    DLOGITS,
    T: tl.constexpr,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    score = tl.load(LOGITS + token * E + expert, expert < E, other=-float("inf"))
    exponential = tl.exp(score - tl.max(score, axis=0))
    probability = exponential / tl.sum(exponential, axis=0)
    # Requested top-2 route fractions sum to one, including later dropped routes.
    count = tl.load(COUNTS + expert, expert < E, other=0).to(tl.float32)
    fraction = count / (2.0 * T)
    weighted = tl.sum(probability * fraction, axis=0)
    scale: tl.constexpr = E / T
    tl.store(ROW_LOSS + token, scale * weighted)
    # Softmax Jacobian: d(sum_e p_e*f_e)/dz_j = p_j*(f_j-sum_e p_e*f_e).
    gradient = scale * probability * (fraction - weighted)
    tl.store(DLOGITS + token * E + expert, gradient, expert < E)


@triton.jit
def expert_gemm_kernel(
    X,
    W,
    COUNTS,
    OUT,
    CAPACITY: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    APPLY_SILU: tl.constexpr,
):
    expert = tl.program_id(2)
    start = tl.program_id(0) * BM
    rows = start + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    count = tl.minimum(tl.load(COUNTS + expert), CAPACITY)
    acc = tl.zeros((BM, BN), tl.float32)
    # Empty experts and wholly padded tiles skip the expensive reduction.
    if start < count:
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + kk
            x = tl.load(
                X + (expert * CAPACITY + rows[:, None]) * K + k[None, :],
                (rows[:, None] < count) & (k[None, :] < K),
                other=0,
            )
            w = tl.load(
                W + (expert * K + k[:, None]) * N + cols[None, :],
                (k[:, None] < K) & (cols[None, :] < N),
                other=0,
            )
            acc = tl.dot(x, w, acc, input_precision="ieee")
    if APPLY_SILU:
        acc = acc / (1.0 + tl.exp(-acc))
    # Explicitly mask unused rows; all capacity rows are initialized for callers.
    result = tl.where(rows[:, None] < count, acc, 0.0)
    tl.store(
        OUT + (expert * CAPACITY + rows[:, None]) * N + cols[None, :],
        result,
        (rows[:, None] < CAPACITY) & (cols[None, :] < N),
    )


@triton.jit
def pack_rank_routes_kernel(
    X,
    IDS,
    COUNTS,
    SEND_X,
    SEND_EXPERT,
    SEND_ROUTE,
    SLOTS,
    LOCAL_E: tl.constexpr,
    RCAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    token = route // 2
    expert = tl.load(IDS + route)
    owner, local_expert = expert // LOCAL_E, expert % LOCAL_E
    # Reserve a unique slot in the destination rank's fixed-capacity chunk.
    slot = tl.atomic_add(COUNTS + owner, 1)
    tl.store(SLOTS + route, slot)
    accepted = slot < RCAP
    feature = tl.arange(0, BLOCK)
    x = tl.load(X + token * D + feature, feature < D, other=0)
    send_row = owner * RCAP + slot
    tl.store(SEND_X + send_row * D + feature, x, accepted & (feature < D))
    tl.store(SEND_EXPERT + send_row, local_expert, accepted)
    # This identity is useful for inspection; SLOTS is the source's inverse map.
    tl.store(SEND_ROUTE + send_row, route, accepted)


@triton.jit
def combine_rank_returns_kernel(
    RETURNED,
    IDS,
    GATES,
    SLOTS,
    Y,
    LOCAL_E: tl.constexpr,
    RCAP: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.float32)
    for choice in tl.static_range(2):
        route = token * 2 + choice
        owner = tl.load(IDS + route) // LOCAL_E
        slot = tl.load(SLOTS + route)
        gate = tl.load(GATES + route)
        y = tl.load(
            RETURNED + (owner * RCAP + slot) * N + feature,
            (slot < RCAP) & (feature < N),
            other=0,
        )
        result += gate * y
    # One owner per output token, so neither reduction nor gather needs atomics.
    tl.store(Y + token * N + feature, result, feature < N)


@triton.jit
def dispatch_received_kernel(
    RECV_X,
    RECV_EXPERT,
    COUNTS,
    PACKED,
    ROW_SLOTS,
    ECAP: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    expert = tl.load(RECV_EXPERT + row)
    slot = tl.full((), -1, tl.int32)
    if expert >= 0:
        slot = tl.atomic_add(COUNTS + expert, 1)
        feature = tl.arange(0, BLOCK)
        x = tl.load(RECV_X + row * D + feature, feature < D, other=0)
        tl.store(
            PACKED + (expert * ECAP + slot) * D + feature,
            x,
            (slot < ECAP) & (feature < D),
        )
    # Preserve the original receive-row -> expert-slot inverse map.
    tl.store(ROW_SLOTS + row, slot)


@triton.jit
def restore_received_kernel(
    EXPERT_OUT,
    RECV_EXPERT,
    ROW_SLOTS,
    RETURN_SEND,
    ECAP: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    expert, slot = tl.load(RECV_EXPERT + row), tl.load(ROW_SLOTS + row)
    feature = tl.arange(0, BLOCK)
    accepted = (expert >= 0) & (slot >= 0) & (slot < ECAP)
    # Clamp padded metadata for pointer arithmetic; the load is still masked.
    safe_expert, safe_slot = tl.maximum(expert, 0), tl.maximum(slot, 0)
    value = tl.load(
        EXPERT_OUT + (safe_expert * ECAP + safe_slot) * N + feature,
        accepted & (feature < N),
        other=0,
    )
    tl.store(RETURN_SEND + row * N + feature, value, feature < N)
