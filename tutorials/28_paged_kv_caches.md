# 28. Paged KV caches: translate logical tokens through a block table

Previous: [Group-scaled quantization](27_group_scaled_quantization.md). Next: [Split-KV decode](29_split_kv_decode.md). Start of the series: [vector addition](01_vector_addition.md).

This self-contained Markdown notebook extends [cache appends](21_kv_cache_updates.md) and [decode attention](22_decode_attention.md). Run sections 1–8 in order, or run section 9 independently. One CUDA GPU holds contiguous FP32 K/V page pools `(NPAGES,Hkv,P,D)`, query rows `(B,Hq,D)`, and a contiguous INT32 block table `(B,MAX_PAGES)`. Hq is a positive multiple of Hkv, D=1–128, and page size P is a power of two from 1 to 32. Inputs are finite and moderate. This inference example allocates exclusive pages, appends one token per sequence, and attends over valid prefixes.

The commented device code is in [paged_kv_cache.py](../kernels/paged_kv_cache.py). Allocation, writing, publishing lengths, and reading are separate ordered stages. The allocator is a small monotonic free-list example; page reclamation and shared-prefix copy-on-write need additional lifecycle management.

## Concepts: a sequence is logically contiguous, physically scattered

A contiguous cache reserves its maximum length for every sequence. A paged cache instead gives each growing sequence physical pages from a shared pool. Logical token t belongs to logical page `t//P`, at offset `t%P`. A block table translates the logical page into a physical page ID:

$$
p=\operatorname{TABLE}[b,\lfloor t/P\rfloor],\qquad
\operatorname{offset}=((pH_{kv}+h)P+(t\bmod P))D+d.
$$

For P=4 and table `[7,2,9]`, tokens 0–3 live in physical page 7, tokens 4–7 in page 2, and tokens 8–11 in page 9. Consecutive logical pages need not have consecutive addresses. The [PagedAttention paper](https://arxiv.org/abs/2309.06180) introduces this cache organization; the [vLLM kernel guide](https://docs.vllm.ai/en/stable/design/paged_attention/) describes its production attention implementation. Our layout and allocator are deliberately simpler.

### 1. Allocate only when crossing a page boundary

Before appending at position L, a new page is needed when `L%P==0` and that logical page is unassigned. One program per sequence atomically reserves a free-list ticket. The free list contains unique unused physical page IDs, so accepted tickets select disjoint pages. An entry of -1 means unallocated.

The counter starts at zero and increases across appends. Do not reset it while pages remain allocated: doing so would give their IDs to another sequence. Atomics make ticket assignment unique, but which sequence gets which ticket is unspecified.

### 2. Allocation failure is explicit

If the free list is exhausted or the sequence reaches logical capacity `MAX_PAGES*P`, status OK is zero. That sequence's append is skipped and its length stays unchanged. Other sequences can still succeed. The caller must inspect OK before treating the new query as an accepted token.

Failed free-list reservations still consume tickets. This demonstration stops or rebuilds an exhausted pool; it does not replenish the list in place. In a long-running server, the scheduler would manage freeing, synchronization with readers, and admission/retry policy. Never recycle a physical page while an earlier attention reader may still use it.

### 3. Publish lengths after every head finishes its write

```text
reserve pages -> write all new K/V heads -> advance accepted lengths -> decode
```

All four launches run on the same stream. During the append launch, LENGTHS is read-only. A separate kernel advances it after all head programs finish. Otherwise, a head that runs late could read an already-incremented length and write a different slot. A length store inside one program is not a synchronization barrier for other programs.

### 4. Attention translates one page at a time

Each query program loads a physical page ID, reads its valid K/V tokens, and updates the online-softmax state from tutorial 22. The last page may be partial. The loop never visits unallocated logical pages. Grouped query heads use `kv_head=query_head//(Hq/Hkv)`. An empty sequence returns zero output and LSE=-infinity as an explicit inference convention; nonempty sequences use ordinary softmax.

Writable pages are exclusively owned here. Prefix sharing would require read-only references, reference counts, and copy-on-write when appending into a shared partial page. Absolute RoPE positions still refer to logical token positions, not physical page IDs.

## 1. Imports and the logical-to-physical address example

```python
import torch
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


page_size = 4
example_table = [7, 2, 9]
for token in (0, 3, 4, 8):
    print(f"logical token {token} -> page {example_table[token // page_size]}, offset {token % page_size}")
```

Page IDs change the address, while token order remains unchanged.

## 2. PyTorch eager: attend to an ordinary dense prefix

```python
tiny_q = torch.tensor([1., -1.], device=DEVICE)
tiny_k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], device=DEVICE)
tiny_v = torch.tensor([[2., 3.], [4., 5.], [6., 7.]], device=DEVICE)
tiny_scores = (tiny_k @ tiny_q) / 2**0.5
print("Dense-prefix output:", (tiny_scores.softmax(0) @ tiny_v).cpu())
```

Paging must preserve this mathematical result. It changes where the prefix is stored.

## 3. PyTorch reference: decode a ragged dense cache

```python
def dense_decode_torch(q, kc, vc, lengths):
    batch, hq, d = q.shape
    hkv = kc.shape[1]
    out = torch.zeros_like(q)
    lse = torch.full((batch, hq), -float("inf"), device=q.device)
    for b, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        for h in range(hq):
            kh = h // (hq // hkv)
            scores = (kc[b, kh, :length] @ q[b, h]) * d**-0.5
            out[b, h] = scores.softmax(0) @ vc[b, kh, :length]
            lse[b, h] = scores.logsumexp(0)
    return out, lse
```

This independent reference sees dense chronological tokens and does not use the block table.

## 4. Triton: allocation, page writes, length publication, and reads

```python
@triton.jit
def allocate_pages_kernel(
    TABLE,
    LENGTHS,
    FREE,
    CURSOR,
    OK,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    NFREE: tl.constexpr,
):
    batch = tl.program_id(0)
    position = tl.load(LENGTHS + batch)
    accepted = tl.full((), 0, tl.int32)
    if position < MAX_PAGES * P:
        logical = position // P
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        # Allocate only at page boundaries and only for an unassigned entry.
        if (position % P == 0) & (page < 0):
            ticket = tl.atomic_add(CURSOR, 1)
            page = tl.load(FREE + ticket, ticket < NFREE, other=-1)
            tl.store(TABLE + batch * MAX_PAGES + logical, page)
        accepted = (page >= 0).to(tl.int32)
    # Every batch item gets a status, even when no new page was needed.
    tl.store(OK + batch, accepted)


@triton.jit
def paged_append_kernel(
    KC,
    VC,
    NK,
    NV,
    TABLE,
    LENGTHS,
    OK,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HKV, row % HKV
    # LENGTHS stays immutable during this launch: every head reads the same slot.
    # Advance lengths in a separate kernel, avoiding a cross-head read/write race.
    position = tl.load(LENGTHS + batch)
    accepted = tl.load(OK + batch) != 0
    page = tl.load(TABLE + batch * MAX_PAGES + position // P, accepted, other=0)
    feature = tl.arange(0, BD)
    offset = ((page * HKV + head) * P + position % P) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, accepted & (feature < D))
    tl.store(VC + offset, v, accepted & (feature < D))


@triton.jit
def advance_lengths_kernel(LENGTHS, OK, B: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + batch, batch < B, other=0)
    accepted = tl.load(OK + batch, batch < B, other=0)
    tl.store(LENGTHS + batch, length + accepted, batch < B)


@triton.jit
def paged_decode_kernel(
    Q,
    KC,
    VC,
    TABLE,
    LENGTHS,
    OUT,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, P)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    for logical in range(tl.cdiv(length, P)):
        # A logical consecutive page may be anywhere in physical storage.
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        t = logical * P + token
        offset = ((page * HKV + kv_head) * P + token[:, None]) * D + feature[None, :]
        valid = (t[:, None] < length) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < length, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    # The loop is skipped for empty sequences; use a safe denominator.
    out = numerator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + row * D + feature, out, feature < D)
    tl.store(LSE + row, maximum + tl.log(tl.where(denominator > 0, denominator, 1.0)))
```

All append programs read the same old lengths. The separate length kernel is part of the required launch sequence.

## 5. Python wrappers: construct an owned pool and launch stages

```python
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def make_cache(length_values, hkv, d, max_pages, page_size, npages):
    if (not length_values or min(hkv, d, max_pages, npages) < 1 or d > 128
        or page_size not in (1, 2, 4, 8, 16, 32)):
        raise ValueError("require positive dimensions, D<=128, and supported page size")
    batch = len(length_values)
    capacity = max_pages * page_size
    if any(length < 0 or length > capacity for length in length_values):
        raise ValueError("lengths must be in [0, logical capacity]")
    if max(npages*hkv*page_size*d, batch*hkv*capacity*d, batch*max_pages) >= 2**31:
        raise ValueError("require 32-bit-safe offsets")
    needed = sum(triton.cdiv(length, page_size) for length in length_values)
    if needed > npages:
        raise ValueError("initial prefixes need more pages than the physical pool")
    # Random physical order makes accidental contiguous addressing visible.
    physical_order = torch.randperm(npages).tolist()
    table = torch.full((batch, max_pages), -1, device=DEVICE, dtype=torch.int32)
    kc = torch.zeros((npages, hkv, page_size, d), device=DEVICE)
    vc = torch.zeros_like(kc)
    dense_k = torch.zeros((batch, hkv, capacity, d), device=DEVICE)
    dense_v = torch.zeros_like(dense_k)
    used = 0
    for b, length in enumerate(length_values):
        dense_k[b, :, :length] = torch.randn(hkv, length, d, device=DEVICE)
        dense_v[b, :, :length] = torch.randn(hkv, length, d, device=DEVICE)
        for logical in range(triton.cdiv(length, page_size)):
            page = physical_order[used]
            used += 1
            table[b, logical] = page
            begin, end = logical*page_size, min((logical+1)*page_size, length)
            kc[page, :, :end-begin] = dense_k[b, :, begin:end]
            vc[page, :, :end-begin] = dense_v[b, :, begin:end]
    state = dict(kc=kc, vc=vc, table=table,
                 lengths=torch.tensor(length_values, device=DEVICE, dtype=torch.int32),
                 free=torch.tensor(physical_order[used:], device=DEVICE, dtype=torch.int32),
                 cursor=torch.zeros(1, device=DEVICE, dtype=torch.int32),
                 ok=torch.empty(batch, device=DEVICE, dtype=torch.int32))
    return state, dense_k, dense_v


def append_paged_triton(state, new_k, new_v):
    require_fp32(state["kc"], state["vc"], new_k, new_v)
    _, hkv, p, d = state["kc"].shape
    batch, max_pages = state["table"].shape
    if new_k.shape != (batch, hkv, d) or new_v.shape != new_k.shape:
        raise ValueError("new K/V must be (B,Hkv,D)")
    # State was initialized by make_cache; launches preserve its ownership invariants.
    allocate_pages_kernel[(batch,)](state["table"], state["lengths"], state["free"],
        state["cursor"], state["ok"], max_pages, p, state["free"].numel(), num_warps=4)
    paged_append_kernel[(batch*hkv,)](state["kc"], state["vc"], new_k, new_v,
        state["table"], state["lengths"], state["ok"], hkv, max_pages, p, d,
        triton.next_power_of_2(d), num_warps=4)
    advance_lengths_kernel[(triton.cdiv(batch, 128),)](state["lengths"], state["ok"], batch, 128)
    return state["ok"].clone()


def paged_decode_triton(q, state):
    require_fp32(q, state["kc"], state["vc"])
    if q.ndim != 3:
        raise ValueError("Q must be (B,Hq,D)")
    batch, hq, d = q.shape
    _, hkv, p, cache_d = state["kc"].shape
    if (batch != state["table"].shape[0] or d != cache_d or hq < 1 or hq % hkv):
        raise ValueError("require matching B/D and Hq a positive multiple of Hkv")
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    paged_decode_kernel[(batch*hq,)](q, state["kc"], state["vc"], state["table"],
        state["lengths"], out, lse, hq, hkv, state["table"].shape[1], p, d,
        triton.next_power_of_2(d), num_warps=4)
    return out, lse


state, dense_k, dense_v = make_cache([0, 3], 2, 7, 3, 2, 8)
query = torch.randn(2, 4, 7, device=DEVICE)
```

These wrappers accept only state constructed by `make_cache`. An external scheduler must separately validate page ownership, free-list uniqueness, valid prefixes, and stream ordering. Do not pass cache views as new K/V buffers.

## 6. Check ragged prefixes, page boundaries, and allocation failure

```python
def check_decode(q, state, dense_k, dense_v):
    out, lse = paged_decode_triton(q, state)
    expected, expected_lse = dense_decode_torch(q, dense_k, dense_v, state["lengths"])
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)


check_decode(query, state, dense_k, dense_v)  # Includes one empty sequence.
for _ in range(4):
    old_lengths = state["lengths"].cpu().tolist()
    new_k = torch.randn(2, 2, 7, device=DEVICE)
    new_v = torch.randn_like(new_k)
    ok = append_paged_triton(state, new_k, new_v).cpu().tolist()
    for b, accepted in enumerate(ok):
        if accepted:
            dense_k[b, :, old_lengths[b]] = new_k[b]
            dense_v[b, :, old_lengths[b]] = new_v[b]
    assert state["lengths"].cpu().tolist() == [length + accepted for length, accepted in zip(old_lengths, ok)]
    check_decode(query, state, dense_k, dense_v)
# The second sequence reached capacity and its fourth append was rejected.
assert state["lengths"].cpu().tolist() == [4, 6]
assigned = state["table"][state["table"] >= 0]
assert assigned.unique().numel() == assigned.numel()
# Pool exhaustion before logical capacity: a new page is needed but FREE is empty.
small, sk, sv = make_cache([2], 1, 3, 2, 2, 1)
before_k, before_v = small["kc"].clone(), small["vc"].clone()
ok = append_paged_triton(small, torch.ones(1, 1, 3, device=DEVICE), torch.ones(1, 1, 3, device=DEVICE))
assert ok.item() == 0 and small["lengths"].item() == 2
torch.testing.assert_close(small["kc"], before_k, rtol=0, atol=0)
torch.testing.assert_close(small["vc"], before_v, rtol=0, atol=0)
check_decode(torch.randn(1, 2, 3, device=DEVICE), small, sk, sv)
print("Paging, ragged/empty decode, exclusive allocation, and failure checks passed.")
```

Do not compare physical page assignment order across runs: atomic ticket assignment can differ. Compare logical cache contents and outputs.

## 7. Benchmark read-only paged decode

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_state, timed_k, timed_v = make_cache([127, 256], 2, 32, 16, 16, 32)
timed_q = torch.randn(2, 4, 32, device=DEVICE)
check_decode(timed_q, timed_state, timed_k, timed_v)
t = bench_ms(lambda: paged_decode_triton(timed_q, timed_state))
print(f"Paged decode: {t*1000:.3f} us")
```

The state is read-only during timing, so repeated calls see the same cache. Allocation and append are excluded. To time updates, prepare fresh state for each measured update or reset all ownership metadata outside the measured interval; repeated appends eventually fill the pool.

## 8. Inspect physical ownership and logical lengths

```python
print("Logical lengths:", state["lengths"].cpu().tolist())
print("Block table:\n", state["table"].cpu())
print("Free-list tickets consumed:", state["cursor"].item())
```

A sequence with length L needs `ceil(L/P)` logical pages. Its unused space in the last owned page is less than P tokens. The pool and block table remain preallocated; paging allocates logical ownership of existing GPU memory, rather than issuing a GPU memory-allocation API call.

## Try next

Continue with [split-KV decode](29_split_kv_decode.md). To combine paging and split-KV, keep split boundaries in logical token coordinates and perform the same block-table translation inside each chunk.

## 9. Complete code in one block

Run this block independently in a fresh notebook, or save it as a Python script. It combines sections 1–8; the kernel definitions match the linked Triton-only file.

```python
# 1. Imports and the logical-to-physical address example
import torch
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


page_size = 4
example_table = [7, 2, 9]
for token in (0, 3, 4, 8):
    print(f"logical token {token} -> page {example_table[token // page_size]}, offset {token % page_size}")


# 2. PyTorch eager: attend to an ordinary dense prefix
tiny_q = torch.tensor([1., -1.], device=DEVICE)
tiny_k = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], device=DEVICE)
tiny_v = torch.tensor([[2., 3.], [4., 5.], [6., 7.]], device=DEVICE)
tiny_scores = (tiny_k @ tiny_q) / 2**0.5
print("Dense-prefix output:", (tiny_scores.softmax(0) @ tiny_v).cpu())


# 3. PyTorch reference: decode a ragged dense cache
def dense_decode_torch(q, kc, vc, lengths):
    batch, hq, d = q.shape
    hkv = kc.shape[1]
    out = torch.zeros_like(q)
    lse = torch.full((batch, hq), -float("inf"), device=q.device)
    for b, length in enumerate(lengths.cpu().tolist()):
        if length == 0:
            continue
        for h in range(hq):
            kh = h // (hq // hkv)
            scores = (kc[b, kh, :length] @ q[b, h]) * d**-0.5
            out[b, h] = scores.softmax(0) @ vc[b, kh, :length]
            lse[b, h] = scores.logsumexp(0)
    return out, lse


# 4. Triton: allocation, page writes, length publication, and reads
@triton.jit
def allocate_pages_kernel(
    TABLE,
    LENGTHS,
    FREE,
    CURSOR,
    OK,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    NFREE: tl.constexpr,
):
    batch = tl.program_id(0)
    position = tl.load(LENGTHS + batch)
    accepted = tl.full((), 0, tl.int32)
    if position < MAX_PAGES * P:
        logical = position // P
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        # Allocate only at page boundaries and only for an unassigned entry.
        if (position % P == 0) & (page < 0):
            ticket = tl.atomic_add(CURSOR, 1)
            page = tl.load(FREE + ticket, ticket < NFREE, other=-1)
            tl.store(TABLE + batch * MAX_PAGES + logical, page)
        accepted = (page >= 0).to(tl.int32)
    # Every batch item gets a status, even when no new page was needed.
    tl.store(OK + batch, accepted)


@triton.jit
def paged_append_kernel(
    KC,
    VC,
    NK,
    NV,
    TABLE,
    LENGTHS,
    OK,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HKV, row % HKV
    # LENGTHS stays immutable during this launch: every head reads the same slot.
    # Advance lengths in a separate kernel, avoiding a cross-head read/write race.
    position = tl.load(LENGTHS + batch)
    accepted = tl.load(OK + batch) != 0
    page = tl.load(TABLE + batch * MAX_PAGES + position // P, accepted, other=0)
    feature = tl.arange(0, BD)
    offset = ((page * HKV + head) * P + position % P) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, accepted & (feature < D))
    tl.store(VC + offset, v, accepted & (feature < D))


@triton.jit
def advance_lengths_kernel(LENGTHS, OK, B: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + batch, batch < B, other=0)
    accepted = tl.load(OK + batch, batch < B, other=0)
    tl.store(LENGTHS + batch, length + accepted, batch < B)


@triton.jit
def paged_decode_kernel(
    Q,
    KC,
    VC,
    TABLE,
    LENGTHS,
    OUT,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, P)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    for logical in range(tl.cdiv(length, P)):
        # A logical consecutive page may be anywhere in physical storage.
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        t = logical * P + token
        offset = ((page * HKV + kv_head) * P + token[:, None]) * D + feature[None, :]
        valid = (t[:, None] < length) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < length, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    # The loop is skipped for empty sequences; use a safe denominator.
    out = numerator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + row * D + feature, out, feature < D)
    tl.store(LSE + row, maximum + tl.log(tl.where(denominator > 0, denominator, 1.0)))


# 5. Python wrappers: construct an owned pool and launch stages
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def make_cache(length_values, hkv, d, max_pages, page_size, npages):
    if (not length_values or min(hkv, d, max_pages, npages) < 1 or d > 128
        or page_size not in (1, 2, 4, 8, 16, 32)):
        raise ValueError("require positive dimensions, D<=128, and supported page size")
    batch = len(length_values)
    capacity = max_pages * page_size
    if any(length < 0 or length > capacity for length in length_values):
        raise ValueError("lengths must be in [0, logical capacity]")
    if max(npages*hkv*page_size*d, batch*hkv*capacity*d, batch*max_pages) >= 2**31:
        raise ValueError("require 32-bit-safe offsets")
    needed = sum(triton.cdiv(length, page_size) for length in length_values)
    if needed > npages:
        raise ValueError("initial prefixes need more pages than the physical pool")
    # Random physical order makes accidental contiguous addressing visible.
    physical_order = torch.randperm(npages).tolist()
    table = torch.full((batch, max_pages), -1, device=DEVICE, dtype=torch.int32)
    kc = torch.zeros((npages, hkv, page_size, d), device=DEVICE)
    vc = torch.zeros_like(kc)
    dense_k = torch.zeros((batch, hkv, capacity, d), device=DEVICE)
    dense_v = torch.zeros_like(dense_k)
    used = 0
    for b, length in enumerate(length_values):
        dense_k[b, :, :length] = torch.randn(hkv, length, d, device=DEVICE)
        dense_v[b, :, :length] = torch.randn(hkv, length, d, device=DEVICE)
        for logical in range(triton.cdiv(length, page_size)):
            page = physical_order[used]
            used += 1
            table[b, logical] = page
            begin, end = logical*page_size, min((logical+1)*page_size, length)
            kc[page, :, :end-begin] = dense_k[b, :, begin:end]
            vc[page, :, :end-begin] = dense_v[b, :, begin:end]
    state = dict(kc=kc, vc=vc, table=table,
                 lengths=torch.tensor(length_values, device=DEVICE, dtype=torch.int32),
                 free=torch.tensor(physical_order[used:], device=DEVICE, dtype=torch.int32),
                 cursor=torch.zeros(1, device=DEVICE, dtype=torch.int32),
                 ok=torch.empty(batch, device=DEVICE, dtype=torch.int32))
    return state, dense_k, dense_v


def append_paged_triton(state, new_k, new_v):
    require_fp32(state["kc"], state["vc"], new_k, new_v)
    _, hkv, p, d = state["kc"].shape
    batch, max_pages = state["table"].shape
    if new_k.shape != (batch, hkv, d) or new_v.shape != new_k.shape:
        raise ValueError("new K/V must be (B,Hkv,D)")
    # State was initialized by make_cache; launches preserve its ownership invariants.
    allocate_pages_kernel[(batch,)](state["table"], state["lengths"], state["free"],
        state["cursor"], state["ok"], max_pages, p, state["free"].numel(), num_warps=4)
    paged_append_kernel[(batch*hkv,)](state["kc"], state["vc"], new_k, new_v,
        state["table"], state["lengths"], state["ok"], hkv, max_pages, p, d,
        triton.next_power_of_2(d), num_warps=4)
    advance_lengths_kernel[(triton.cdiv(batch, 128),)](state["lengths"], state["ok"], batch, 128)
    return state["ok"].clone()


def paged_decode_triton(q, state):
    require_fp32(q, state["kc"], state["vc"])
    if q.ndim != 3:
        raise ValueError("Q must be (B,Hq,D)")
    batch, hq, d = q.shape
    _, hkv, p, cache_d = state["kc"].shape
    if (batch != state["table"].shape[0] or d != cache_d or hq < 1 or hq % hkv):
        raise ValueError("require matching B/D and Hq a positive multiple of Hkv")
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    paged_decode_kernel[(batch*hq,)](q, state["kc"], state["vc"], state["table"],
        state["lengths"], out, lse, hq, hkv, state["table"].shape[1], p, d,
        triton.next_power_of_2(d), num_warps=4)
    return out, lse


state, dense_k, dense_v = make_cache([0, 3], 2, 7, 3, 2, 8)
query = torch.randn(2, 4, 7, device=DEVICE)


# 6. Check ragged prefixes, page boundaries, and allocation failure
def check_decode(q, state, dense_k, dense_v):
    out, lse = paged_decode_triton(q, state)
    expected, expected_lse = dense_decode_torch(q, dense_k, dense_v, state["lengths"])
    torch.testing.assert_close(out, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=2e-4, atol=2e-4)


check_decode(query, state, dense_k, dense_v)  # Includes one empty sequence.
for _ in range(4):
    old_lengths = state["lengths"].cpu().tolist()
    new_k = torch.randn(2, 2, 7, device=DEVICE)
    new_v = torch.randn_like(new_k)
    ok = append_paged_triton(state, new_k, new_v).cpu().tolist()
    for b, accepted in enumerate(ok):
        if accepted:
            dense_k[b, :, old_lengths[b]] = new_k[b]
            dense_v[b, :, old_lengths[b]] = new_v[b]
    assert state["lengths"].cpu().tolist() == [length + accepted for length, accepted in zip(old_lengths, ok)]
    check_decode(query, state, dense_k, dense_v)
# The second sequence reached capacity and its fourth append was rejected.
assert state["lengths"].cpu().tolist() == [4, 6]
assigned = state["table"][state["table"] >= 0]
assert assigned.unique().numel() == assigned.numel()
# Pool exhaustion before logical capacity: a new page is needed but FREE is empty.
small, sk, sv = make_cache([2], 1, 3, 2, 2, 1)
before_k, before_v = small["kc"].clone(), small["vc"].clone()
ok = append_paged_triton(small, torch.ones(1, 1, 3, device=DEVICE), torch.ones(1, 1, 3, device=DEVICE))
assert ok.item() == 0 and small["lengths"].item() == 2
torch.testing.assert_close(small["kc"], before_k, rtol=0, atol=0)
torch.testing.assert_close(small["vc"], before_v, rtol=0, atol=0)
check_decode(torch.randn(1, 2, 3, device=DEVICE), small, sk, sv)
print("Paging, ragged/empty decode, exclusive allocation, and failure checks passed.")


# 7. Benchmark read-only paged decode
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_state, timed_k, timed_v = make_cache([127, 256], 2, 32, 16, 16, 32)
timed_q = torch.randn(2, 4, 32, device=DEVICE)
check_decode(timed_q, timed_state, timed_k, timed_v)
t = bench_ms(lambda: paged_decode_triton(timed_q, timed_state))
print(f"Paged decode: {t*1000:.3f} us")


# 8. Inspect physical ownership and logical lengths
print("Logical lengths:", state["lengths"].cpu().tolist())
print("Block table:\n", state["table"].cpu())
print("Free-list tickets consumed:", state["cursor"].item())
```
