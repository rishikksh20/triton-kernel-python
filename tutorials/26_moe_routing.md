# 26. MoE routing: deterministic top-2 choices and capacity-limited scatter

Previous: [Cross entropy](25_cross_entropy.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Routing logits are finite, moderate, contiguous FP32 `(T,E)` and token features are contiguous FP32 `(T,D)` on one NVIDIA CUDA GPU. T >= 1, E=2–256, and D=1–1024. This forward-only example selects exactly two distinct experts per token, normalizes their selected logits, dispatches to per-expert buffers, and combines a toy expert result. There is no registered backward, routing noise, auxiliary loss, or distributed communication.

## Concepts: select experts, assign slots, then scatter

For token t, choose expert indices e_0,e_1 with the two largest routing logits. Normalize **over the two selected experts**:

$$
w_{t,j}=\frac{e^{z_{t,e_j}-m_t}}{e^{z_{t,e_0}-m_t}+e^{z_{t,e_1}-m_t}},\qquad
y_t=\sum_{j=0}^1 w_{t,j}\,\operatorname{Expert}_{e_j}(x_t).
$$

The [sparsely gated MoE paper](https://arxiv.org/abs/1701.06538) motivates conditional computation through a router. This notebook isolates a simple top-2 routing/dispatch mechanism; real models choose their own gate normalization, capacity, balancing, and expert schedules.

### 1. A tie rule makes the router reproducible

Find the maximum logit, then take the smallest expert index among equal maxima. Mask that index and repeat. Padded experts use negative infinity and cannot win. This avoids relying on unspecified tie ordering in a top-k API. With equal logits, the first two experts are 0 and 1.

### 2. Compact expert slots require coordination

Each selected route atomically increments its expert counter and receives the previous count as a unique slot. If the slot is below capacity C, write the token into `(E,C,D)` storage. Record every route's slot, including overflow slots, so the combine stage knows which routes were retained.

```text
logits -> top-2 expert IDs + weights
IDs -> atomic expert counter -> unique slot
token -> scatter into expert buffer if slot < capacity
expert outputs + slots + weights -> gather and sum per token
```

### 3. Overflow is a policy, not a pointer error

Counters report total requested routes. Stored routes equal `min(count,C)` for each expert. Overflow routes are dropped, contribute zero, and the surviving weights are **not renormalized** in this guide. Under overflow, the set of retained tokens can vary with GPU scheduling. Without overflow, buffer order can still vary; compare token membership and reconstructed outputs rather than physical row order.

### 4. Reset mutable state on every dispatch

Counters start at zero, token IDs at -1, and packed feature buffers at zero. Repeated timing must repeat that initialization; reusing incremented counters measures progressively overflowing workloads. A later expert kernel must only process populated slots or handle padded ones explicitly. We use a toy linear expert scale to verify scatter/gather before adding expensive expert GEMMs.

The combine kernel has one owner per output token, so its two expert contributions need no output atomics. Routing/scatter and combine kernels on the same stream execute in order.

## 1. Imports and a router with deliberate ties

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
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))

logits = torch.tensor([[2., 2., 0.], [-1., 3., 1.], [0., 0., 0.]], device=DEVICE)
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=DEVICE)
```

The first and last rows test tie-breaking independently of random data.

## 2. PyTorch eager: select and normalize two routes

```python
ids_eager = logits.argsort(dim=-1, descending=True, stable=True)[:, :2]
selected = logits.gather(1, ids_eager)
gates_eager = selected.softmax(dim=-1)
print("Expert IDs:\n", ids_eager.cpu())
print("Selected-expert weights:\n", gates_eager.cpu())
```

Stable descending sort keeps the lower original expert index first on ties.

## 3. PyTorch function: an independent deterministic routing reference

```python
def route_torch(logits):
    ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :2]
    gates = logits.gather(1, ids).softmax(dim=-1)
    return ids.to(torch.int32), gates
```

This reference sorts all experts. It provides the intended tie contract; it is not an optimized top-2 router.

## 4. Triton: top-2 reduction, atomic dispatch, and token-owned combine

```python
@triton.jit
def route_kernel(Z, IDS, GATES, E: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    score = tl.load(Z + token * E + expert, expert < E, other=-float("inf"))
    first_score = tl.max(score, axis=0)
    first = tl.min(tl.where((expert < E) & (score == first_score), expert, 2147483647), axis=0)
    remaining = tl.where(expert != first, score, -float("inf"))
    second_score = tl.max(remaining, axis=0)
    second = tl.min(tl.where((expert < E) & (expert != first) & (remaining == second_score),
                             expert, 2147483647), axis=0)
    ratio = tl.exp(second_score - first_score)
    first_gate = 1 / (1 + ratio)
    second_gate = ratio / (1 + ratio)
    tl.store(IDS + token * 2, first)
    tl.store(IDS + token * 2 + 1, second)
    tl.store(GATES + token * 2, first_gate)
    tl.store(GATES + token * 2 + 1, second_gate)


@triton.jit
def dispatch_kernel(X, IDS, COUNTS, PACKED, TOKENS, SLOTS,
                    CAPACITY: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0)
    token = route // 2
    expert = tl.load(IDS + route)
    slot = tl.atomic_add(COUNTS + expert, 1)
    tl.store(SLOTS + route, slot)
    feature = tl.arange(0, BLOCK)
    values = tl.load(X + token * D + feature, feature < D, other=0)
    accepted = slot < CAPACITY
    tl.store(PACKED + (expert * CAPACITY + slot) * D + feature,
             values, accepted & (feature < D))
    tl.store(TOKENS + expert * CAPACITY + slot, token, accepted)


@triton.jit
def combine_kernel(EXPERT_OUT, IDS, GATES, SLOTS, Y,
                   CAPACITY: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.float32)
    for choice in tl.static_range(2):
        route = token * 2 + choice
        expert = tl.load(IDS + route)
        slot = tl.load(SLOTS + route)
        gate = tl.load(GATES + route)
        values = tl.load(EXPERT_OUT + (expert * CAPACITY + slot) * D + feature,
                         (slot < CAPACITY) & (feature < D), other=0)
        result += gate * values
    tl.store(Y + token * D + feature, result, feature < D)
```

Atomic counters coordinate compact slots, and masks prevent overflow writes/reads. Returning original token IDs makes buffer membership inspectable.

## 5. Python wrappers: validate metadata and initialize dispatch buffers

```python
def route_triton(logits):
    if logits.ndim != 2 or logits.shape[0] < 1 or not 2 <= logits.shape[1] <= 256:
        raise ValueError("require logits (T,E), T >= 1, E=2..256")
    if logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous() or logits.numel() >= 2**31:
        raise ValueError("logits must be contiguous CUDA FP32 with 32-bit-safe offsets")
    tokens, experts = logits.shape
    ids = torch.empty((tokens, 2), device=logits.device, dtype=torch.int32)
    gates = torch.empty((tokens, 2), device=logits.device, dtype=torch.float32)
    route_kernel[(tokens,)](logits, ids, gates, experts, triton.next_power_of_2(experts), num_warps=4)
    return ids, gates


def dispatch_prevalidated(x, ids, experts, capacity):
    tokens, d = x.shape
    counts = torch.zeros(experts, device=x.device, dtype=torch.int32)
    packed = torch.zeros((experts, capacity, d), device=x.device, dtype=torch.float32)
    token_ids = torch.full((experts, capacity), -1, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    dispatch_kernel[(tokens*2,)](x, ids, counts, packed, token_ids, slots,
                                 capacity, d, triton.next_power_of_2(d), num_warps=4)
    return dict(counts=counts, packed=packed, tokens=token_ids, slots=slots)


def dispatch_triton(x, ids, experts, capacity):
    if x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 1024:
        raise ValueError("require features (T,D), T >= 1, D=1..1024")
    if ids.shape != (x.shape[0], 2) or ids.dtype != torch.int32:
        raise ValueError("ids must be INT32 (T,2)")
    if not 2 <= experts <= 256 or capacity < 1 or max(x.numel(), experts*capacity*x.shape[1], x.shape[0]*2) >= 2**31:
        raise ValueError("invalid expert count/capacity or unsafe offsets/counters")
    if x.dtype != torch.float32 or not x.is_cuda or ids.device != x.device or not x.is_contiguous() or not ids.is_contiguous():
        raise ValueError("features and IDs must be contiguous on the same CUDA device")
    if ((ids < 0) | (ids >= experts)).any().item() or (ids[:, 0] == ids[:, 1]).any().item():
        raise ValueError("each token must select two distinct in-range experts")
    return dispatch_prevalidated(x, ids, experts, capacity)


def combine_triton(expert_out, ids, gates, slots):
    # This launch helper consumes validated outputs of route/dispatch and matching expert storage.
    experts, capacity, d = expert_out.shape
    out = torch.empty((ids.shape[0], d), device=expert_out.device, dtype=torch.float32)
    combine_kernel[(ids.shape[0],)](expert_out, ids, gates, slots, out,
                                   capacity, d, triton.next_power_of_2(d), num_warps=4)
    return out
```

IDs are validated before dispatch, which synchronizes. `dispatch_prevalidated` still resets all mutable buffers on every call. The combine helper assumes matching contiguous buffers from the earlier steps.

## 6. Check routing, unordered dispatch membership, and reconstruction

```python
ids, gates = route_triton(logits)
ref_ids, ref_gates = route_torch(logits)
torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
for tokens, experts, d in ((1, 2, 1), (17, 3, 7), (33, 17, 48), (7, 256, 128)):
    scores = torch.randn(tokens, experts, device=DEVICE)
    scores[0] = 0.
    features = torch.randn(tokens, d, device=DEVICE)
    ids, gates = route_triton(scores)
    ref_ids, ref_gates = route_torch(scores)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
    dispatch = dispatch_triton(features, ids, experts, capacity=tokens)
    expected_counts = torch.bincount(ids.long().flatten(), minlength=experts).to(torch.int32)
    torch.testing.assert_close(dispatch["counts"], expected_counts, rtol=0, atol=0)
    for expert in range(experts):
        count = expected_counts[expert].item()
        stored_tokens = dispatch["tokens"][expert, :count].long()
        expected_tokens = (ids == expert).any(dim=1).nonzero().flatten()
        torch.testing.assert_close(stored_tokens.sort().values, expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(dispatch["packed"][expert, :count], features[stored_tokens], rtol=0, atol=0)
    expert_scale = torch.arange(1, experts + 1, device=DEVICE).float()[:, None, None]
    expert_out = dispatch["packed"] * expert_scale
    actual = combine_triton(expert_out, ids, gates, dispatch["slots"])
    expected = features * ((ids.float() + 1) * gates).sum(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Tie rules, gates, counters, unordered membership, and reconstruction passed.")
```

Capacity T guarantees no overflow because each token selects an expert at most once. The toy expert multiplies features by expert_index+1, making the expected combined result independent of buffer order.

## 7. Benchmark routing and dispatch with counters reset each time

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

scores = torch.randn(256, 16, device=DEVICE)
features = torch.randn(256, 64, device=DEVICE)
ids, gates = route_triton(scores)
ref_ids, ref_gates = route_torch(scores)
torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
dispatch_triton(features, ids, 16, 256)  # Validate metadata and ID values once.
p = bench_ms(lambda: route_torch(scores))
t = bench_ms(lambda: route_triton(scores))
scatter_ms = bench_ms(lambda: dispatch_prevalidated(features, ids, 16, 256))
print(f"Routing: stable PyTorch sort {p*1000:.3f} us, Triton top-2 {t*1000:.3f} us")
print(f"Dispatch including allocations/initialization: {scatter_ms*1000:.3f} us")
```

Router timing includes output allocation and excludes expert computation. Dispatch timing includes zeroed counters, zeroed packed storage, token sentinels, and scatter. No initialized-state reuse is hidden in the measurement; checks, compilation, and printing are outside timing.

## 8. Exercise overflow without depending on scheduling order

```python
scores = torch.zeros(8, 3, device=DEVICE)
features = torch.arange(32, device=DEVICE).float().reshape(8, 4)
ids, gates = route_triton(scores)
capacity = 2
dispatch = dispatch_triton(features, ids, 3, capacity)
torch.testing.assert_close(dispatch["counts"], torch.tensor([8, 8, 0], device=DEVICE, dtype=torch.int32))
for expert in (0, 1):
    retained = dispatch["tokens"][expert].long()
    assert retained.unique().numel() == capacity
    assert ((retained >= 0) & (retained < 8)).all().item()
    torch.testing.assert_close(dispatch["packed"][expert], features[retained], rtol=0, atol=0)
assert (dispatch["tokens"][2] == -1).all().item()
expert_out = dispatch["packed"] * torch.arange(1, 4, device=DEVICE).float()[:, None, None]
actual = combine_triton(expert_out, ids, gates, dispatch["slots"])
accepted = dispatch["slots"] < capacity
expected = features * (((ids.float() + 1) * gates * accepted).sum(1, keepdim=True))
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Overflow drops routes, preserves requested counts, and combines retained routes only.")
```

The two accepted token sets need not match each other or another run. Tests check the declared overflow policy rather than a particular atomic ordering.

## Try next

Replace the toy expert scale with per-expert [batched GEMMs](11_batched_matmul.md) or [SwiGLU](24_swiglu.md), using populated slots only. Then study prefix-sum dispatch, load-balancing losses, and distributed expert communication as separate extensions.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a router with deliberate ties
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))

logits = torch.tensor([[2., 2., 0.], [-1., 3., 1.], [0., 0., 0.]], device=DEVICE)
x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=DEVICE)


# 2. PyTorch eager: select and normalize two routes
ids_eager = logits.argsort(dim=-1, descending=True, stable=True)[:, :2]
selected = logits.gather(1, ids_eager)
gates_eager = selected.softmax(dim=-1)
print("Expert IDs:\n", ids_eager.cpu())
print("Selected-expert weights:\n", gates_eager.cpu())


# 3. PyTorch function: an independent deterministic routing reference
def route_torch(logits):
    ids = logits.argsort(dim=-1, descending=True, stable=True)[:, :2]
    gates = logits.gather(1, ids).softmax(dim=-1)
    return ids.to(torch.int32), gates


# 4. Triton: top-2 reduction, atomic dispatch, and token-owned combine
@triton.jit
def route_kernel(Z, IDS, GATES, E: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    score = tl.load(Z + token * E + expert, expert < E, other=-float("inf"))
    first_score = tl.max(score, axis=0)
    first = tl.min(tl.where((expert < E) & (score == first_score), expert, 2147483647), axis=0)
    remaining = tl.where(expert != first, score, -float("inf"))
    second_score = tl.max(remaining, axis=0)
    second = tl.min(tl.where((expert < E) & (expert != first) & (remaining == second_score),
                             expert, 2147483647), axis=0)
    ratio = tl.exp(second_score - first_score)
    first_gate = 1 / (1 + ratio)
    second_gate = ratio / (1 + ratio)
    tl.store(IDS + token * 2, first)
    tl.store(IDS + token * 2 + 1, second)
    tl.store(GATES + token * 2, first_gate)
    tl.store(GATES + token * 2 + 1, second_gate)


@triton.jit
def dispatch_kernel(X, IDS, COUNTS, PACKED, TOKENS, SLOTS,
                    CAPACITY: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    route = tl.program_id(0)
    token = route // 2
    expert = tl.load(IDS + route)
    slot = tl.atomic_add(COUNTS + expert, 1)
    tl.store(SLOTS + route, slot)
    feature = tl.arange(0, BLOCK)
    values = tl.load(X + token * D + feature, feature < D, other=0)
    accepted = slot < CAPACITY
    tl.store(PACKED + (expert * CAPACITY + slot) * D + feature,
             values, accepted & (feature < D))
    tl.store(TOKENS + expert * CAPACITY + slot, token, accepted)


@triton.jit
def combine_kernel(EXPERT_OUT, IDS, GATES, SLOTS, Y,
                   CAPACITY: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    feature = tl.arange(0, BLOCK)
    result = tl.zeros((BLOCK,), tl.float32)
    for choice in tl.static_range(2):
        route = token * 2 + choice
        expert = tl.load(IDS + route)
        slot = tl.load(SLOTS + route)
        gate = tl.load(GATES + route)
        values = tl.load(EXPERT_OUT + (expert * CAPACITY + slot) * D + feature,
                         (slot < CAPACITY) & (feature < D), other=0)
        result += gate * values
    tl.store(Y + token * D + feature, result, feature < D)


# 5. Python wrappers: validate metadata and initialize dispatch buffers
def route_triton(logits):
    if logits.ndim != 2 or logits.shape[0] < 1 or not 2 <= logits.shape[1] <= 256:
        raise ValueError("require logits (T,E), T >= 1, E=2..256")
    if logits.dtype != torch.float32 or not logits.is_cuda or not logits.is_contiguous() or logits.numel() >= 2**31:
        raise ValueError("logits must be contiguous CUDA FP32 with 32-bit-safe offsets")
    tokens, experts = logits.shape
    ids = torch.empty((tokens, 2), device=logits.device, dtype=torch.int32)
    gates = torch.empty((tokens, 2), device=logits.device, dtype=torch.float32)
    route_kernel[(tokens,)](logits, ids, gates, experts, triton.next_power_of_2(experts), num_warps=4)
    return ids, gates


def dispatch_prevalidated(x, ids, experts, capacity):
    tokens, d = x.shape
    counts = torch.zeros(experts, device=x.device, dtype=torch.int32)
    packed = torch.zeros((experts, capacity, d), device=x.device, dtype=torch.float32)
    token_ids = torch.full((experts, capacity), -1, device=x.device, dtype=torch.int32)
    slots = torch.empty_like(ids)
    dispatch_kernel[(tokens*2,)](x, ids, counts, packed, token_ids, slots,
                                 capacity, d, triton.next_power_of_2(d), num_warps=4)
    return dict(counts=counts, packed=packed, tokens=token_ids, slots=slots)


def dispatch_triton(x, ids, experts, capacity):
    if x.ndim != 2 or x.shape[0] < 1 or not 1 <= x.shape[1] <= 1024:
        raise ValueError("require features (T,D), T >= 1, D=1..1024")
    if ids.shape != (x.shape[0], 2) or ids.dtype != torch.int32:
        raise ValueError("ids must be INT32 (T,2)")
    if not 2 <= experts <= 256 or capacity < 1 or max(x.numel(), experts*capacity*x.shape[1], x.shape[0]*2) >= 2**31:
        raise ValueError("invalid expert count/capacity or unsafe offsets/counters")
    if x.dtype != torch.float32 or not x.is_cuda or ids.device != x.device or not x.is_contiguous() or not ids.is_contiguous():
        raise ValueError("features and IDs must be contiguous on the same CUDA device")
    if ((ids < 0) | (ids >= experts)).any().item() or (ids[:, 0] == ids[:, 1]).any().item():
        raise ValueError("each token must select two distinct in-range experts")
    return dispatch_prevalidated(x, ids, experts, capacity)


def combine_triton(expert_out, ids, gates, slots):
    # This launch helper consumes validated outputs of route/dispatch and matching expert storage.
    experts, capacity, d = expert_out.shape
    out = torch.empty((ids.shape[0], d), device=expert_out.device, dtype=torch.float32)
    combine_kernel[(ids.shape[0],)](expert_out, ids, gates, slots, out,
                                   capacity, d, triton.next_power_of_2(d), num_warps=4)
    return out


# 6. Check routing, unordered dispatch membership, and reconstruction
ids, gates = route_triton(logits)
ref_ids, ref_gates = route_torch(logits)
torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
for tokens, experts, d in ((1, 2, 1), (17, 3, 7), (33, 17, 48), (7, 256, 128)):
    scores = torch.randn(tokens, experts, device=DEVICE)
    scores[0] = 0.
    features = torch.randn(tokens, d, device=DEVICE)
    ids, gates = route_triton(scores)
    ref_ids, ref_gates = route_torch(scores)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
    dispatch = dispatch_triton(features, ids, experts, capacity=tokens)
    expected_counts = torch.bincount(ids.long().flatten(), minlength=experts).to(torch.int32)
    torch.testing.assert_close(dispatch["counts"], expected_counts, rtol=0, atol=0)
    for expert in range(experts):
        count = expected_counts[expert].item()
        stored_tokens = dispatch["tokens"][expert, :count].long()
        expected_tokens = (ids == expert).any(dim=1).nonzero().flatten()
        torch.testing.assert_close(stored_tokens.sort().values, expected_tokens, rtol=0, atol=0)
        torch.testing.assert_close(dispatch["packed"][expert, :count], features[stored_tokens], rtol=0, atol=0)
    expert_scale = torch.arange(1, experts + 1, device=DEVICE).float()[:, None, None]
    expert_out = dispatch["packed"] * expert_scale
    actual = combine_triton(expert_out, ids, gates, dispatch["slots"])
    expected = features * ((ids.float() + 1) * gates).sum(dim=1, keepdim=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Tie rules, gates, counters, unordered membership, and reconstruction passed.")


# 7. Benchmark routing and dispatch with counters reset each time
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

scores = torch.randn(256, 16, device=DEVICE)
features = torch.randn(256, 64, device=DEVICE)
ids, gates = route_triton(scores)
ref_ids, ref_gates = route_torch(scores)
torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
torch.testing.assert_close(gates, ref_gates, rtol=1e-5, atol=1e-6)
dispatch_triton(features, ids, 16, 256)  # Validate metadata and ID values once.
p = bench_ms(lambda: route_torch(scores))
t = bench_ms(lambda: route_triton(scores))
scatter_ms = bench_ms(lambda: dispatch_prevalidated(features, ids, 16, 256))
print(f"Routing: stable PyTorch sort {p*1000:.3f} us, Triton top-2 {t*1000:.3f} us")
print(f"Dispatch including allocations/initialization: {scatter_ms*1000:.3f} us")


# 8. Exercise overflow without depending on scheduling order
scores = torch.zeros(8, 3, device=DEVICE)
features = torch.arange(32, device=DEVICE).float().reshape(8, 4)
ids, gates = route_triton(scores)
capacity = 2
dispatch = dispatch_triton(features, ids, 3, capacity)
torch.testing.assert_close(dispatch["counts"], torch.tensor([8, 8, 0], device=DEVICE, dtype=torch.int32))
for expert in (0, 1):
    retained = dispatch["tokens"][expert].long()
    assert retained.unique().numel() == capacity
    assert ((retained >= 0) & (retained < 8)).all().item()
    torch.testing.assert_close(dispatch["packed"][expert], features[retained], rtol=0, atol=0)
assert (dispatch["tokens"][2] == -1).all().item()
expert_out = dispatch["packed"] * torch.arange(1, 4, device=DEVICE).float()[:, None, None]
actual = combine_triton(expert_out, ids, gates, dispatch["slots"])
accepted = dispatch["slots"] < capacity
expected = features * (((ids.float() + 1) * gates * accepted).sum(1, keepdim=True))
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Overflow drops routes, preserves requested counts, and combines retained routes only.")
```
