# 31. Expert computation: per-expert GEMMs, load balance, and distributed routing

Previous: [Fused training operations](30_fused_training_operations.md). Start of the series: [vector addition](01_vector_addition.md).

This self-contained Markdown notebook replaces tutorial 26's toy expert scaling with two-layer experts, then extends routing to owner ranks. Run sections 1–8 in order, or section 9 independently. Contiguous FP32 X is `(T,D)`, router logits are `(T,E)`, up weights are `(E,D,H)`, and down weights are `(E,H,N)`. Dimensions are positive, E=2–256, D/N=1–1024, values are finite and moderate, and buffer element counts are below `2**31`. Each token chooses two distinct experts with selected-logit softmax gates. Experts compute `SiLU(X@Wup[e])@Wdown[e]`, without biases. This example has no registered autograd backward; it also demonstrates an optional auxiliary router loss with a raw logits gradient.

The commented new device code is in [expert_computation.py](../kernels/expert_computation.py), reusing [moe_routing.py](../kernels/moe_routing.py). Section 8 includes a runnable two-rank simulation on one GPU and an optional real collective function for initialized distributed workers. Triton handles packing, dispatch, expert GEMMs, restoring row order, and combining. Host collectives transport buffers between GPUs.

## Concepts: turn routed tokens into matrix rows

### 1. Each expert receives a different number of rows

Tutorial 26 assigns each accepted route an `(expert,slot)` pair. The packed input is `(E,CAPACITY,D)`. For expert e, valid rows are `0 <= slot < min(COUNTS[e],CAPACITY)`. Run one tiled GEMM for its up projection, apply SiLU after the complete D reduction, then run a down GEMM over H. The weighted combine gathers two results into each original token.

$$
Z_e=\operatorname{SiLU}(X_eW^{up}_e),\qquad
O_e=Z_eW^{down}_e,\qquad
Y_t=\sum_{j=0}^1 g_{t,j}O_{e_{t,j},slot_{t,j}}.
$$

The launch grid is `(row_tiles,column_tiles,experts)`. Each program selects its expert's weight matrix. Counts mask unused rows. Wholly unused row tiles skip the dot loop, and every padded output row is explicitly written as zero. This fixed rectangular grid is easy to inspect; a compact work list or persistent tile scheduler could avoid launching padded programs altogether.

### 2. Capacity is not load balancing

Counters report requested routes, including dropped ones. A capacity cap limits storage and computation but does not redirect routes to underused experts. Overflow routes contribute zero, and surviving gate weights are not renormalized, matching tutorial 26. With distinct top-2 IDs, CAPACITY=T guarantees every expert has enough local slots.

Useful diagnostics are requested count per expert, retained count, dropped count, and `max(count)/mean(count)`. A skewed router can leave most expert programs empty while one expert dominates work. Training can encourage balanced routing; inference schedules can group active tiles more efficiently. These are separate choices from the mathematical expert function. The [Switch Transformer paper](https://arxiv.org/abs/2101.03961) discusses expert capacity and balancing in sparse expert models; this guide retains tutorial 26's top-2 gate contract rather than implementing Switch's top-1 router.

We adapt its balancing objective to two routes per token. Let `f_e=COUNTS[e]/(2*T)` be the requested route fraction, including routes later dropped, and let `P_e=mean_t softmax(logits[t,:])[e]` use the full-expert softmax. Then

$$
L_{aux}=E\sum_e f_e P_e,\qquad
\frac{\partial L_{aux}}{\partial z_{t,e}}=\frac{E}{T}p_{t,e}
    \left(f_e-\sum_j f_jp_{t,j}\right).
$$

The selected top-2 gate weights and full-expert probabilities serve different purposes. Treat the discrete counts as fixed metadata during this derivative. The kernel writes per-token loss contributions and the raw logits gradient; a host reduction sums contributions. Section 7 compares both with PyTorch autograd for balanced and skewed examples. A training system would weight this objective and add its gradient to the task-loss router gradient. It encourages balance across training updates and does not repair overflow within the current batch.

### 3. A global expert ID identifies a rank and a local expert

With WORLD ranks and LOCAL_E experts per rank, contiguous ownership uses

$$
owner=e//LOCAL\_E,\qquad local\_expert=e\bmod LOCAL\_E.
$$

Source programs atomically assign routes to destination-rank slots. The send tensor is `(WORLD,RCAP,D)`: its first axis is destination rank. Parallel metadata contains local expert IDs; -1 marks padding. The source saves each original route's slot and gate.

```text
source tokens -> destination-rank pack -> all-to-all
  -> receiver's per-expert dispatch -> two expert GEMMs
  -> restore receiver row order -> reverse all-to-all
  -> source's saved (owner,slot,gate) -> weighted token combine
```

After the forward exchange, the receiver tensor's first axis is source rank. Receiver dispatch reorders rows into local expert buffers, so it must save an inverse map. Restore outputs to original receive-row order before the reverse exchange. Then each source receives `(owner_rank,original_send_slot,N)` and can gather its own token outputs without output atomics. Route numbers are local to a source; they are not global identities.

### 4. Start with fixed-size communication

Every rank agrees on WORLD, RCAP, D, and N. Each collective sends RCAP rows to every peer, including zero padding. RCAP at least twice the maximum source token count guarantees no source overflow. A receiver expert capacity of `WORLD*RCAP` guarantees no receiver overflow, though it can waste substantial memory. The source and receiver capacities are different limits.

The optional host function uses [PyTorch all_to_all_single](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single). All workers must invoke the same collectives in the same order on correctly configured devices/streams. They exchange features and local expert IDs, then return restored outputs. Triton kernels do not launch network collectives themselves. Uneven packed communication additionally needs count exchange, prefix offsets, matching send/receive split sizes, and corresponding reverse splits.

Fixed capacity, FP32 IEEE dots, and padded exchanges prioritize understanding. They do not establish production throughput or train the router. A production implementation would measure local GEMMs, padding, load imbalance, and communication together.

## 1. Imports and small expert weights

```python
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


tokens, experts, d, hidden, n = 7, 4, 5, 9, 3
x = torch.randn(tokens, d, device=DEVICE)
logits = torch.randn(tokens, experts, device=DEVICE)
wup = torch.randn(experts, d, hidden, device=DEVICE) / d**0.5
wdown = torch.randn(experts, hidden, n, device=DEVICE) / hidden**0.5
```

Different input, hidden, and output widths make it easier to catch mistaken strides.

## 2. PyTorch eager: compute one selected expert result

```python
chosen = logits[0].argsort(descending=True, stable=True)[:2]
chosen_gates = logits[0, chosen].softmax(0)
first_expert = int(chosen[0])
first_value = F.silu(x[0] @ wup[first_expert]) @ wdown[first_expert]
print("Token 0 experts:", chosen.cpu().tolist(), "gates:", chosen_gates.cpu().tolist())
print("Its first expert result:", first_value.cpu())
```

The router chooses weights; expert GEMMs transform features. Gate multiplication happens when combining expert outputs.

## 3. PyTorch reference: direct token-owned expert evaluation

```python
def expert_reference(x, ids, gates, wup, wdown, slots=None, capacity=None):
    result = torch.zeros((x.shape[0], wdown.shape[2]), device=x.device)
    ids_cpu = ids.cpu().tolist()
    slots_cpu = slots.cpu().tolist() if slots is not None else None
    for token, selected in enumerate(ids_cpu):
        for choice, expert in enumerate(selected):
            if slots_cpu is not None and slots_cpu[token][choice] >= capacity:
                continue
            value = F.silu(x[token] @ wup[expert]) @ wdown[expert]
            result[token] += gates[token, choice] * value
    return result
```

The reference computes original token features directly. Overflow comparisons use the actual retained-route mask because GPU atomic scheduling can choose different winners.

## 4. Triton: route, dispatch, compute, communicate buffers, and combine

The first three kernels are the existing tutorial 26 kernels, included here to keep the notebook self-contained. The remaining kernels match the new raw file.

```python
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
```

Per-expert GEMMs use an expert grid axis. Rank buffers and received-row inverse maps carry the routing information through communication.

## 5. Python wrappers: local MoE and rank-owned buffers

```python
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def compute_experts_prevalidated(packed, counts, wup, wdown):
    experts, capacity, d = packed.shape
    hidden, n = wup.shape[2], wdown.shape[2]
    intermediate = torch.empty((experts, capacity, hidden), device=packed.device)
    output = torch.empty((experts, capacity, n), device=packed.device)
    expert_gemm_kernel[(triton.cdiv(capacity, 32), triton.cdiv(hidden, 32), experts)](
        packed, wup, counts, intermediate, capacity, d, hidden, 32, 32, 32, True, num_warps=4)
    expert_gemm_kernel[(triton.cdiv(capacity, 32), triton.cdiv(n, 32), experts)](
        intermediate, wdown, counts, output, capacity, hidden, n, 32, 32, 32, False, num_warps=4)
    return output


def local_moe_prevalidated(x, logits, wup, wdown, capacity):
    tokens, d = x.shape
    experts = logits.shape[1]
    n = wdown.shape[2]
    ids = torch.empty((tokens, 2), device=x.device, dtype=torch.int32)
    gates = torch.empty((tokens, 2), device=x.device)
    route_kernel[(tokens,)](logits, ids, gates, experts, triton.next_power_of_2(experts), num_warps=4)
    counts = torch.zeros(experts, device=x.device, dtype=torch.int32)
    packed = torch.zeros((experts, capacity, d), device=x.device)
    token_ids = torch.full((experts, capacity), -1, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    dispatch_kernel[(tokens*2,)](x, ids, counts, packed, token_ids, slots,
        capacity, d, triton.next_power_of_2(d), num_warps=4)
    expert_out = compute_experts_prevalidated(packed, counts, wup, wdown)
    y = torch.empty((tokens, n), device=x.device)
    combine_kernel[(tokens,)](expert_out, ids, gates, slots, y,
        capacity, n, triton.next_power_of_2(n), num_warps=4)
    return dict(y=y, ids=ids, gates=gates, slots=slots, counts=counts,
                packed=packed, token_ids=token_ids, expert_out=expert_out)


def local_moe_triton(x, logits, wup, wdown, capacity=None):
    require_fp32(x, logits, wup, wdown)
    if x.ndim != 2 or logits.ndim != 2 or wup.ndim != 3 or wdown.ndim != 3:
        raise ValueError("require X/logits matrices and per-expert rank-three weights")
    tokens, d = x.shape
    experts, wd, hidden = wup.shape
    n = wdown.shape[2]
    if (tokens < 1 or not 2 <= experts <= 256 or not 1 <= d <= 1024 or not 1 <= n <= 1024
        or hidden < 1 or wd != d or logits.shape != (tokens, experts)
        or wdown.shape[:2] != (experts, hidden)):
        raise ValueError("incompatible positive expert dimensions")
    capacity = tokens if capacity is None else capacity
    if (not isinstance(capacity, int) or capacity < 1 or tokens*2 >= 2**31
        or max(experts*capacity*max(d, hidden, n), tokens*n) >= 2**31):
        raise ValueError("require positive capacity and 32-bit-safe offsets/counters")
    return local_moe_prevalidated(x, logits, wup, wdown, capacity)


def load_balance_triton(logits, counts):
    require_fp32(logits)
    if logits.ndim != 2 or logits.shape[0] < 1 or not 2 <= logits.shape[1] <= 256:
        raise ValueError("require nonempty logits (T,E), E=2..256")
    tokens, experts = logits.shape
    if (counts.dtype != torch.int32 or counts.device != logits.device
        or counts.shape != (experts,) or not counts.is_contiguous()):
        raise ValueError("counts must be contiguous INT32 (E,) on the logits device")
    if bool((counts < 0).any()) or counts.long().sum().item() != 2*tokens:
        raise ValueError("counts must describe all requested top-2 routes")
    row_loss = torch.empty(tokens, device=logits.device)
    dlogits = torch.empty_like(logits)
    load_balance_kernel[(tokens,)](logits, counts, row_loss, dlogits,
        tokens, experts, triton.next_power_of_2(experts), num_warps=4)
    return row_loss.sum(), dlogits


def pack_owners_prevalidated(x, ids, world, local_experts, rank_capacity):
    tokens, d = x.shape
    send_x = torch.zeros((world, rank_capacity, d), device=x.device)
    send_expert = torch.full((world, rank_capacity), -1, device=x.device, dtype=torch.int32)
    send_route = torch.full_like(send_expert, -1)
    counts = torch.zeros(world, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    pack_rank_routes_kernel[(tokens*2,)](x, ids, counts, send_x, send_expert, send_route, slots,
        local_experts, rank_capacity, d, triton.next_power_of_2(d), num_warps=4)
    return dict(x=send_x, expert=send_expert, route=send_route, slots=slots, counts=counts)


def compute_received_prevalidated(recv_x, recv_expert, wup_local, wdown_local, expert_capacity):
    world, rank_capacity, d = recv_x.shape
    local_experts, n = wup_local.shape[0], wdown_local.shape[2]
    counts = torch.zeros(local_experts, device=recv_x.device, dtype=torch.int32)
    packed = torch.zeros((local_experts, expert_capacity, d), device=recv_x.device)
    row_slots = torch.empty_like(recv_expert)
    dispatch_received_kernel[(world*rank_capacity,)](recv_x, recv_expert, counts, packed,
        row_slots, expert_capacity, d, triton.next_power_of_2(d), num_warps=4)
    expert_out = compute_experts_prevalidated(packed, counts, wup_local, wdown_local)
    restored = torch.empty((world, rank_capacity, n), device=recv_x.device)
    restore_received_kernel[(world*rank_capacity,)](expert_out, recv_expert, row_slots, restored,
        expert_capacity, n, triton.next_power_of_2(n), num_warps=4)
    return restored


def combine_returns_prevalidated(returned, ids, gates, slots, local_experts):
    tokens, n = ids.shape[0], returned.shape[2]
    rank_capacity = returned.shape[1]
    y = torch.empty((tokens, n), device=returned.device)
    combine_rank_returns_kernel[(tokens,)](returned, ids, gates, slots, y,
        local_experts, rank_capacity, n, triton.next_power_of_2(n), num_warps=4)
    return y


result = local_moe_triton(x, logits, wup, wdown)
```

Rank helpers are explicitly prevalidated. Their callers must enforce same-device contiguous buffers, in-range IDs, positive matching capacities/dimensions, and 32-bit-safe scratch/counter sizes. The local wrapper validates its public tensor contract and resets counters on every dispatch.

## 6. Check direct expert results, empty experts, and overflow

```python
expected = expert_reference(x, result["ids"], result["gates"], wup, wdown)
torch.testing.assert_close(result["y"], expected, rtol=2e-4, atol=2e-4)
stable_ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :2].to(torch.int32)
torch.testing.assert_close(result["ids"], stable_ids, rtol=0, atol=0)
torch.testing.assert_close(result["gates"], logits.gather(1, stable_ids.long()).softmax(-1))
for t, d, h, n in ((1, 1, 1, 1), (33, 17, 35, 19)):
    sample_x = torch.randn(t, d, device=DEVICE)
    # Tied logits select experts 0 and 1, leaving 2 and 3 empty.
    tied = torch.zeros(t, 4, device=DEVICE)
    up_weights = torch.randn(4, d, h, device=DEVICE) / d**0.5
    down_weights = torch.randn(4, h, n, device=DEVICE) / h**0.5
    for capacity in (t, 1):
        actual = local_moe_triton(sample_x, tied, up_weights, down_weights, capacity)
        expected = expert_reference(sample_x, actual["ids"], actual["gates"],
            up_weights, down_weights, actual["slots"], capacity)
        torch.testing.assert_close(actual["y"], expected, rtol=3e-4, atol=3e-4)
        assert actual["counts"].cpu().tolist() == [t, t, 0, 0]
        assert torch.count_nonzero(actual["expert_out"][2:]).item() == 0
print("Per-expert GEMMs, ties, empty experts, tails, and dropped-route checks passed.")
```

Requested counts can exceed capacity. Only retained slots feed expert computation and the combine; padded outputs are explicitly zero.

## 7. Benchmark local routing plus both expert projections

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_x = torch.randn(128, 64, device=DEVICE)
timed_logits = torch.randn(128, 4, device=DEVICE)
timed_up = torch.randn(4, 64, 96, device=DEVICE) / 64**0.5
timed_down = torch.randn(4, 96, 32, device=DEVICE) / 96**0.5
timed = local_moe_triton(timed_x, timed_logits, timed_up, timed_down)
expected = expert_reference(timed_x, timed["ids"], timed["gates"], timed_up, timed_down)
torch.testing.assert_close(timed["y"], expected, rtol=3e-4, atol=3e-4)
full_ms = bench_ms(lambda: local_moe_prevalidated(timed_x, timed_logits, timed_up, timed_down, 128))
gemm_ms = bench_ms(lambda: compute_experts_prevalidated(timed["packed"], timed["counts"], timed_up, timed_down))
print(f"Local route/dispatch/GEMMs/combine {full_ms*1000:.3f} us; expert GEMMs {gemm_ms*1000:.3f} us")
requested = timed["counts"].float()
retained = requested.clamp_max(128)
print("Requested:", requested.cpu().tolist(), "retained:", retained.cpu().tolist())
print("Max/mean route load:", (requested.max()/requested.mean()).item())
print("Dropped routes:", (requested-retained).sum().item())
# Compare the auxiliary objective and its derivative with autograd.
balanced_logits = torch.tensor([[3., 2., -3., -4.], [-3., -4., 3., 2.]], device=DEVICE).repeat(64, 1)
skewed_logits = balanced_logits[:1].repeat(128, 1)
for label, router_logits in (("balanced", balanced_logits), ("skewed", skewed_logits)):
    routed = local_moe_triton(timed_x, router_logits, timed_up, timed_down)
    aux, dz = load_balance_triton(router_logits, routed["counts"])
    leaf = router_logits.clone().requires_grad_()
    fraction = routed["counts"].float() / (2*leaf.shape[0])
    expected_aux = leaf.shape[1] * (leaf.softmax(-1).mean(0) * fraction).sum()
    expected_dz, = torch.autograd.grad(expected_aux, leaf)
    torch.testing.assert_close(aux, expected_aux.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(dz, expected_dz, rtol=2e-5, atol=2e-6)
    print(label, "route counts:", routed["counts"].cpu().tolist(), "balance loss:", aux.item())
```

The full path allocates/initializes buffers and resets counters on every call. Expert-only timing uses the same fixed packed inputs and counts. Neither timing includes cross-GPU communication or router-logit computation.

## 8. Run a two-rank round trip, then inspect the real collective boundary

The simulation uses two source batches on one GPU. It permutes rank chunks exactly as the two collective exchanges would, while running all dispatch, GEMM, restoration, and combine kernels in Triton.

```python
def simulate_two_ranks(xs, ids_list, gates_list, wup, wdown, rank_capacity):
    world, local_experts = 2, wup.shape[0] // 2
    assert wup.shape[0] % 2 == 0 and len(xs) == len(ids_list) == len(gates_list) == 2
    sends = [pack_owners_prevalidated(xs[s], ids_list[s], world, local_experts, rank_capacity)
             for s in range(world)]
    return_sends = []
    for owner in range(world):
        # First exchange: at the receiver, axis 0 denotes SOURCE rank.
        recv_x = torch.stack([sends[source]["x"][owner] for source in range(world)])
        recv_expert = torch.stack([sends[source]["expert"][owner] for source in range(world)])
        begin, end = owner*local_experts, (owner+1)*local_experts
        return_sends.append(compute_received_prevalidated(recv_x, recv_expert,
            wup[begin:end].contiguous(), wdown[begin:end].contiguous(), world*rank_capacity))
    outputs = []
    for source in range(world):
        # Reverse exchange: at the source, axis 0 denotes original OWNER rank.
        returned = torch.stack([return_sends[owner][source] for owner in range(world)])
        outputs.append(combine_returns_prevalidated(returned, ids_list[source], gates_list[source],
                                                   sends[source]["slots"], local_experts))
    return outputs, sends


x_other = torch.randn_like(x)
logits_other = torch.randn_like(logits)
other = local_moe_triton(x_other, logits_other, wup, wdown)
ids_list, gates_list = [result["ids"], other["ids"]], [result["gates"], other["gates"]]
for rank_capacity in (tokens*2, 1):
    outputs, sends = simulate_two_ranks([x, x_other], ids_list, gates_list, wup, wdown, rank_capacity)
    for source, source_x in enumerate((x, x_other)):
        expected = expert_reference(source_x, ids_list[source], gates_list[source], wup, wdown,
                                    sends[source]["slots"], rank_capacity)
        torch.testing.assert_close(outputs[source], expected, rtol=3e-4, atol=3e-4)
        for owner in range(2):
            valid = sends[source]["route"][owner] >= 0
            sent_routes = sends[source]["route"][owner, valid].long()
            assert torch.equal(ids_list[source].flatten()[sent_routes] // 2,
                               torch.full_like(sent_routes, owner, dtype=torch.int32))
print("Two-rank simulation and rank-capacity overflow checks passed.")


# Optional: invoke on EVERY initialized distributed worker, with its local expert weights.
# This function is defined here; the single-GPU notebook does not call it.
def distributed_experts_prevalidated(x, ids, gates, wup_local, wdown_local, rank_capacity):
    import torch.distributed as dist
    world = dist.get_world_size()
    local_experts = wup_local.shape[0]
    sends = pack_owners_prevalidated(x, ids, world, local_experts, rank_capacity)
    recv_x = torch.empty_like(sends["x"])
    recv_expert = torch.empty_like(sends["expert"])
    # WORLD equal chunks on dimension 0, identical RCAP/D on all workers.
    dist.all_to_all_single(recv_x, sends["x"])
    dist.all_to_all_single(recv_expert, sends["expert"])
    return_send = compute_received_prevalidated(recv_x, recv_expert, wup_local, wdown_local,
                                               world*rank_capacity)
    returned = torch.empty_like(return_send)
    dist.all_to_all_single(returned, return_send)
    y = combine_returns_prevalidated(returned, ids, gates, sends["slots"], local_experts)
    return y, sends["counts"]
```

The optional function requires a process group with a backend that supports these collectives, one correctly selected CUDA device per worker, agreed shapes/capacities, validated global expert IDs, and the same collective order everywhere. Load the contiguous local expert slice for each owner rank. Test against the single-GPU direct reference before timing multiple GPUs. It uses blocking collectives on the current stream; explicit completion/events are needed if adding asynchronous work or other streams. For smaller capacity, rank overflow is dropped without gate renormalization. Empty rank chunks remain initialized zero padding with expert ID -1.

## Try next

You have reached the end of the 31-step sequence. Try replacing the fixed expert grid with a compact active-tile work list, compare balanced and skewed routing, and add count-based uneven all-to-all buffers. Keep the received-row inverse mapping intact when changing the schedule.

## 9. Complete code in one block

Run this block independently in a fresh notebook, or save it as a Python script. It combines sections 1–8; the kernel definitions match the linked Triton-only file.

```python
# 1. Imports and small expert weights
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


tokens, experts, d, hidden, n = 7, 4, 5, 9, 3
x = torch.randn(tokens, d, device=DEVICE)
logits = torch.randn(tokens, experts, device=DEVICE)
wup = torch.randn(experts, d, hidden, device=DEVICE) / d**0.5
wdown = torch.randn(experts, hidden, n, device=DEVICE) / hidden**0.5


# 2. PyTorch eager: compute one selected expert result
chosen = logits[0].argsort(descending=True, stable=True)[:2]
chosen_gates = logits[0, chosen].softmax(0)
first_expert = int(chosen[0])
first_value = F.silu(x[0] @ wup[first_expert]) @ wdown[first_expert]
print("Token 0 experts:", chosen.cpu().tolist(), "gates:", chosen_gates.cpu().tolist())
print("Its first expert result:", first_value.cpu())


# 3. PyTorch reference: direct token-owned expert evaluation
def expert_reference(x, ids, gates, wup, wdown, slots=None, capacity=None):
    result = torch.zeros((x.shape[0], wdown.shape[2]), device=x.device)
    ids_cpu = ids.cpu().tolist()
    slots_cpu = slots.cpu().tolist() if slots is not None else None
    for token, selected in enumerate(ids_cpu):
        for choice, expert in enumerate(selected):
            if slots_cpu is not None and slots_cpu[token][choice] >= capacity:
                continue
            value = F.silu(x[token] @ wup[expert]) @ wdown[expert]
            result[token] += gates[token, choice] * value
    return result


# 4. Triton: route, dispatch, compute, communicate buffers, and combine
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


# 5. Python wrappers: local MoE and rank-owned buffers
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def compute_experts_prevalidated(packed, counts, wup, wdown):
    experts, capacity, d = packed.shape
    hidden, n = wup.shape[2], wdown.shape[2]
    intermediate = torch.empty((experts, capacity, hidden), device=packed.device)
    output = torch.empty((experts, capacity, n), device=packed.device)
    expert_gemm_kernel[(triton.cdiv(capacity, 32), triton.cdiv(hidden, 32), experts)](
        packed, wup, counts, intermediate, capacity, d, hidden, 32, 32, 32, True, num_warps=4)
    expert_gemm_kernel[(triton.cdiv(capacity, 32), triton.cdiv(n, 32), experts)](
        intermediate, wdown, counts, output, capacity, hidden, n, 32, 32, 32, False, num_warps=4)
    return output


def local_moe_prevalidated(x, logits, wup, wdown, capacity):
    tokens, d = x.shape
    experts = logits.shape[1]
    n = wdown.shape[2]
    ids = torch.empty((tokens, 2), device=x.device, dtype=torch.int32)
    gates = torch.empty((tokens, 2), device=x.device)
    route_kernel[(tokens,)](logits, ids, gates, experts, triton.next_power_of_2(experts), num_warps=4)
    counts = torch.zeros(experts, device=x.device, dtype=torch.int32)
    packed = torch.zeros((experts, capacity, d), device=x.device)
    token_ids = torch.full((experts, capacity), -1, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    dispatch_kernel[(tokens*2,)](x, ids, counts, packed, token_ids, slots,
        capacity, d, triton.next_power_of_2(d), num_warps=4)
    expert_out = compute_experts_prevalidated(packed, counts, wup, wdown)
    y = torch.empty((tokens, n), device=x.device)
    combine_kernel[(tokens,)](expert_out, ids, gates, slots, y,
        capacity, n, triton.next_power_of_2(n), num_warps=4)
    return dict(y=y, ids=ids, gates=gates, slots=slots, counts=counts,
                packed=packed, token_ids=token_ids, expert_out=expert_out)


def local_moe_triton(x, logits, wup, wdown, capacity=None):
    require_fp32(x, logits, wup, wdown)
    if x.ndim != 2 or logits.ndim != 2 or wup.ndim != 3 or wdown.ndim != 3:
        raise ValueError("require X/logits matrices and per-expert rank-three weights")
    tokens, d = x.shape
    experts, wd, hidden = wup.shape
    n = wdown.shape[2]
    if (tokens < 1 or not 2 <= experts <= 256 or not 1 <= d <= 1024 or not 1 <= n <= 1024
        or hidden < 1 or wd != d or logits.shape != (tokens, experts)
        or wdown.shape[:2] != (experts, hidden)):
        raise ValueError("incompatible positive expert dimensions")
    capacity = tokens if capacity is None else capacity
    if (not isinstance(capacity, int) or capacity < 1 or tokens*2 >= 2**31
        or max(experts*capacity*max(d, hidden, n), tokens*n) >= 2**31):
        raise ValueError("require positive capacity and 32-bit-safe offsets/counters")
    return local_moe_prevalidated(x, logits, wup, wdown, capacity)


def load_balance_triton(logits, counts):
    require_fp32(logits)
    if logits.ndim != 2 or logits.shape[0] < 1 or not 2 <= logits.shape[1] <= 256:
        raise ValueError("require nonempty logits (T,E), E=2..256")
    tokens, experts = logits.shape
    if (counts.dtype != torch.int32 or counts.device != logits.device
        or counts.shape != (experts,) or not counts.is_contiguous()):
        raise ValueError("counts must be contiguous INT32 (E,) on the logits device")
    if bool((counts < 0).any()) or counts.long().sum().item() != 2*tokens:
        raise ValueError("counts must describe all requested top-2 routes")
    row_loss = torch.empty(tokens, device=logits.device)
    dlogits = torch.empty_like(logits)
    load_balance_kernel[(tokens,)](logits, counts, row_loss, dlogits,
        tokens, experts, triton.next_power_of_2(experts), num_warps=4)
    return row_loss.sum(), dlogits


def pack_owners_prevalidated(x, ids, world, local_experts, rank_capacity):
    tokens, d = x.shape
    send_x = torch.zeros((world, rank_capacity, d), device=x.device)
    send_expert = torch.full((world, rank_capacity), -1, device=x.device, dtype=torch.int32)
    send_route = torch.full_like(send_expert, -1)
    counts = torch.zeros(world, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    pack_rank_routes_kernel[(tokens*2,)](x, ids, counts, send_x, send_expert, send_route, slots,
        local_experts, rank_capacity, d, triton.next_power_of_2(d), num_warps=4)
    return dict(x=send_x, expert=send_expert, route=send_route, slots=slots, counts=counts)


def compute_received_prevalidated(recv_x, recv_expert, wup_local, wdown_local, expert_capacity):
    world, rank_capacity, d = recv_x.shape
    local_experts, n = wup_local.shape[0], wdown_local.shape[2]
    counts = torch.zeros(local_experts, device=recv_x.device, dtype=torch.int32)
    packed = torch.zeros((local_experts, expert_capacity, d), device=recv_x.device)
    row_slots = torch.empty_like(recv_expert)
    dispatch_received_kernel[(world*rank_capacity,)](recv_x, recv_expert, counts, packed,
        row_slots, expert_capacity, d, triton.next_power_of_2(d), num_warps=4)
    expert_out = compute_experts_prevalidated(packed, counts, wup_local, wdown_local)
    restored = torch.empty((world, rank_capacity, n), device=recv_x.device)
    restore_received_kernel[(world*rank_capacity,)](expert_out, recv_expert, row_slots, restored,
        expert_capacity, n, triton.next_power_of_2(n), num_warps=4)
    return restored


def combine_returns_prevalidated(returned, ids, gates, slots, local_experts):
    tokens, n = ids.shape[0], returned.shape[2]
    rank_capacity = returned.shape[1]
    y = torch.empty((tokens, n), device=returned.device)
    combine_rank_returns_kernel[(tokens,)](returned, ids, gates, slots, y,
        local_experts, rank_capacity, n, triton.next_power_of_2(n), num_warps=4)
    return y


result = local_moe_triton(x, logits, wup, wdown)


# 6. Check direct expert results, empty experts, and overflow
expected = expert_reference(x, result["ids"], result["gates"], wup, wdown)
torch.testing.assert_close(result["y"], expected, rtol=2e-4, atol=2e-4)
stable_ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :2].to(torch.int32)
torch.testing.assert_close(result["ids"], stable_ids, rtol=0, atol=0)
torch.testing.assert_close(result["gates"], logits.gather(1, stable_ids.long()).softmax(-1))
for t, d, h, n in ((1, 1, 1, 1), (33, 17, 35, 19)):
    sample_x = torch.randn(t, d, device=DEVICE)
    # Tied logits select experts 0 and 1, leaving 2 and 3 empty.
    tied = torch.zeros(t, 4, device=DEVICE)
    up_weights = torch.randn(4, d, h, device=DEVICE) / d**0.5
    down_weights = torch.randn(4, h, n, device=DEVICE) / h**0.5
    for capacity in (t, 1):
        actual = local_moe_triton(sample_x, tied, up_weights, down_weights, capacity)
        expected = expert_reference(sample_x, actual["ids"], actual["gates"],
            up_weights, down_weights, actual["slots"], capacity)
        torch.testing.assert_close(actual["y"], expected, rtol=3e-4, atol=3e-4)
        assert actual["counts"].cpu().tolist() == [t, t, 0, 0]
        assert torch.count_nonzero(actual["expert_out"][2:]).item() == 0
print("Per-expert GEMMs, ties, empty experts, tails, and dropped-route checks passed.")


# 7. Benchmark local routing plus both expert projections
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_x = torch.randn(128, 64, device=DEVICE)
timed_logits = torch.randn(128, 4, device=DEVICE)
timed_up = torch.randn(4, 64, 96, device=DEVICE) / 64**0.5
timed_down = torch.randn(4, 96, 32, device=DEVICE) / 96**0.5
timed = local_moe_triton(timed_x, timed_logits, timed_up, timed_down)
expected = expert_reference(timed_x, timed["ids"], timed["gates"], timed_up, timed_down)
torch.testing.assert_close(timed["y"], expected, rtol=3e-4, atol=3e-4)
full_ms = bench_ms(lambda: local_moe_prevalidated(timed_x, timed_logits, timed_up, timed_down, 128))
gemm_ms = bench_ms(lambda: compute_experts_prevalidated(timed["packed"], timed["counts"], timed_up, timed_down))
print(f"Local route/dispatch/GEMMs/combine {full_ms*1000:.3f} us; expert GEMMs {gemm_ms*1000:.3f} us")
requested = timed["counts"].float()
retained = requested.clamp_max(128)
print("Requested:", requested.cpu().tolist(), "retained:", retained.cpu().tolist())
print("Max/mean route load:", (requested.max()/requested.mean()).item())
print("Dropped routes:", (requested-retained).sum().item())
# Compare the auxiliary objective and its derivative with autograd.
balanced_logits = torch.tensor([[3., 2., -3., -4.], [-3., -4., 3., 2.]], device=DEVICE).repeat(64, 1)
skewed_logits = balanced_logits[:1].repeat(128, 1)
for label, router_logits in (("balanced", balanced_logits), ("skewed", skewed_logits)):
    routed = local_moe_triton(timed_x, router_logits, timed_up, timed_down)
    aux, dz = load_balance_triton(router_logits, routed["counts"])
    leaf = router_logits.clone().requires_grad_()
    fraction = routed["counts"].float() / (2*leaf.shape[0])
    expected_aux = leaf.shape[1] * (leaf.softmax(-1).mean(0) * fraction).sum()
    expected_dz, = torch.autograd.grad(expected_aux, leaf)
    torch.testing.assert_close(aux, expected_aux.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(dz, expected_dz, rtol=2e-5, atol=2e-6)
    print(label, "route counts:", routed["counts"].cpu().tolist(), "balance loss:", aux.item())


# 8. Run a two-rank round trip, then inspect the real collective boundary
def simulate_two_ranks(xs, ids_list, gates_list, wup, wdown, rank_capacity):
    world, local_experts = 2, wup.shape[0] // 2
    assert wup.shape[0] % 2 == 0 and len(xs) == len(ids_list) == len(gates_list) == 2
    sends = [pack_owners_prevalidated(xs[s], ids_list[s], world, local_experts, rank_capacity)
             for s in range(world)]
    return_sends = []
    for owner in range(world):
        # First exchange: at the receiver, axis 0 denotes SOURCE rank.
        recv_x = torch.stack([sends[source]["x"][owner] for source in range(world)])
        recv_expert = torch.stack([sends[source]["expert"][owner] for source in range(world)])
        begin, end = owner*local_experts, (owner+1)*local_experts
        return_sends.append(compute_received_prevalidated(recv_x, recv_expert,
            wup[begin:end].contiguous(), wdown[begin:end].contiguous(), world*rank_capacity))
    outputs = []
    for source in range(world):
        # Reverse exchange: at the source, axis 0 denotes original OWNER rank.
        returned = torch.stack([return_sends[owner][source] for owner in range(world)])
        outputs.append(combine_returns_prevalidated(returned, ids_list[source], gates_list[source],
                                                   sends[source]["slots"], local_experts))
    return outputs, sends


x_other = torch.randn_like(x)
logits_other = torch.randn_like(logits)
other = local_moe_triton(x_other, logits_other, wup, wdown)
ids_list, gates_list = [result["ids"], other["ids"]], [result["gates"], other["gates"]]
for rank_capacity in (tokens*2, 1):
    outputs, sends = simulate_two_ranks([x, x_other], ids_list, gates_list, wup, wdown, rank_capacity)
    for source, source_x in enumerate((x, x_other)):
        expected = expert_reference(source_x, ids_list[source], gates_list[source], wup, wdown,
                                    sends[source]["slots"], rank_capacity)
        torch.testing.assert_close(outputs[source], expected, rtol=3e-4, atol=3e-4)
        for owner in range(2):
            valid = sends[source]["route"][owner] >= 0
            sent_routes = sends[source]["route"][owner, valid].long()
            assert torch.equal(ids_list[source].flatten()[sent_routes] // 2,
                               torch.full_like(sent_routes, owner, dtype=torch.int32))
print("Two-rank simulation and rank-capacity overflow checks passed.")


# Optional: invoke on EVERY initialized distributed worker, with its local expert weights.
# This function is defined here; the single-GPU notebook does not call it.
def distributed_experts_prevalidated(x, ids, gates, wup_local, wdown_local, rank_capacity):
    import torch.distributed as dist
    world = dist.get_world_size()
    local_experts = wup_local.shape[0]
    sends = pack_owners_prevalidated(x, ids, world, local_experts, rank_capacity)
    recv_x = torch.empty_like(sends["x"])
    recv_expert = torch.empty_like(sends["expert"])
    # WORLD equal chunks on dimension 0, identical RCAP/D on all workers.
    dist.all_to_all_single(recv_x, sends["x"])
    dist.all_to_all_single(recv_expert, sends["expert"])
    return_send = compute_received_prevalidated(recv_x, recv_expert, wup_local, wdown_local,
                                               world*rank_capacity)
    returned = torch.empty_like(return_send)
    dist.all_to_all_single(returned, return_send)
    y = combine_returns_prevalidated(returned, ids, gates, sends["slots"], local_experts)
    return y, sends["counts"]
```
