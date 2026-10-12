# 29. Split-KV decode: parallel cache chunks and mergeable softmax states

Previous: [Paged KV caches](28_paged_kv_caches.md). Next: [Fused training operations](30_fused_training_operations.md). Start of the series: [vector addition](01_vector_addition.md).

This self-contained Markdown notebook extends [online softmax](13_online_softmax.md) and [decode attention](22_decode_attention.md). Run sections 1–8 in order, or section 9 independently. Contiguous FP32 Q is `(B,Hq,D)`; K/V are `(B,Hkv,C,D)`; contiguous INT32 lengths `(B,)` lie in `[1,C]`. Hq is a positive multiple of Hkv, D=1–128, and split count S=1–128. All buffers are on one NVIDIA CUDA GPU with element counts below `2**31`. Values are finite and moderate. This inference example uses a dense cache to isolate the split/merge schedule; paging can be added by changing the cache address calculation.

The commented device code is in [split_kv_decode.py](../kernels/split_kv_decode.py). Each partial program stores an unnormalized softmax state, and a second kernel merges all states for one query.

## Concepts: give one query more parallel work

Tutorial 22 launches one program per batch/query head. That program scans the whole cache prefix. With a small batch and long cache, there may be too few programs to keep the GPU busy. Split-KV launches S programs for each query, each reading a different cache interval.

### 1. Define chunks in cache coordinates

Let `CHUNK=ceil(C/S)`. Split j reads `[j*CHUNK,min((j+1)*CHUNK,L_b))`. Intervals cover every valid token exactly once. Lengths are ragged, so some chunks can be empty; S can even exceed the physical capacity C. Mask the end of each chunk separately from the end of the whole sequence, or tokens near chunk boundaries would be counted twice.

### 2. Store the three quantities that can be merged

For scores s_t within chunk j, store its maximum, shifted denominator, and unnormalized value numerator:

$$
m_j=\max_{t\in j}s_t,\qquad
\ell_j=\sum_{t\in j}e^{s_t-m_j},\qquad
u_j=\sum_{t\in j}e^{s_t-m_j}v_t.
$$

The partial kernel uses the same online recurrence as tutorial 22 while streaming its own chunk. Empty chunks store `m=-inf,l=0,u=0`. Scratch shapes are `(B,Hq,S)` for m/l and `(B,Hq,S,D)` for u. Every scratch entry is written on every launch.

### 3. Shift every state to one common maximum

At merge time, compute

$$
M=\max_{j:\ell_j>0}m_j,\qquad
a_j=\begin{cases}e^{m_j-M}&\ell_j>0\\0&\ell_j=0,\end{cases}
\qquad o=\frac{\sum_j a_ju_j}{\sum_j a_j\ell_j},
\qquad \operatorname{LSE}=M+\log\sum_j a_j\ell_j.
$$

For two states `(m,l,u)`, this is the online-softmax update applied to whole chunks. The [online softmax paper](https://arxiv.org/abs/1805.02867) develops the normalizer recurrence used by the earlier tutorials.

```text
one query -> chunk 0 -> (m0,l0,u0) --+
          -> chunk 1 -> (m1,l1,u1) --+-> common max -> rescaled sums -> output
          -> chunk 2 -> (m2,l2,u2) --+
```

Averaging the normalized chunk outputs `u_j/l_j` is incorrect: chunks have different softmax mass. Even equally sized chunks can have very different score distributions. The merge needs their normalizers as well as their values.

### 4. Handle inactive states before exponentiation

Empty chunks must never evaluate `exp(-inf - -inf)`. Their scan loop is skipped. At merge, at least one chunk is active because L>=1, so M is finite. Replace inactive maxima with M before subtracting, then set their correction to zero. Triton's [where documentation](https://triton-lang.org/main/python-api/generated/triton.language.where.html) specifies that both branches are evaluated; `where` alone does not make an unsafe arithmetic expression lazy.

More splits increase parallelism and shorten each scan, but also add scratch traffic, a second launch, and merge work. Choose S by measuring full partial-plus-merge latency rather than timing only the first stage. Floating-point summation order changes with S, so test with tolerances.

## 1. Imports and ragged grouped-query inputs

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


q = torch.randn(2, 4, 7, device=DEVICE)
kc = torch.randn(2, 2, 70, 7, device=DEVICE)
vc = torch.randn_like(kc)
lengths = torch.tensor([1, 65], device=DEVICE, dtype=torch.int32)
```

The short first sequence makes most splits empty. The second length and head dimension exercise masks.

## 2. PyTorch eager: inspect one query over its prefix

```python
scores = (kc[1, 0, :65] @ q[1, 0]) * 7**-0.5
print("One-head output:", (scores.softmax(0) @ vc[1, 0, :65]).cpu())
print("One-head LSE:", scores.logsumexp(0).item())
```

Both output and LSE should be invariant to splitting, within FP32 rounding error.

## 3. PyTorch reference: independent full-prefix attention

```python
def decode_torch(q, kc, vc, lengths):
    batch, hq, d = q.shape
    hkv = kc.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    for b, length in enumerate(lengths.cpu().tolist()):
        for h in range(hq):
            kh = h // (hq // hkv)
            score = (kc[b, kh, :length] @ q[b, h]) * d**-0.5
            out[b, h] = score.softmax(0) @ vc[b, kh, :length]
            lse[b, h] = score.logsumexp(0)
    return out, lse
```

The reference materializes full-prefix scores and does not share the split/merge algorithm.

## 4. Triton: independent scans and a query-owned merge

```python
@triton.jit
def split_kv_partial_kernel(
    Q,
    KC,
    VC,
    LENGTHS,
    PM,
    PL,
    PU,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, BT)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    begin = split * CHUNK
    end = tl.minimum(begin + CHUNK, length)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    # Empty chunks skip the loop, avoiding exp(-inf - -inf).
    for start in range(begin, end, BT):
        t = start + token
        offset = ((batch * HKV + kv_head) * C + t[:, None]) * D + feature[None, :]
        valid = (t[:, None] < end) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < end, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    state = row * SPLITS + split
    tl.store(PM + state, maximum)
    tl.store(PL + state, denominator)
    tl.store(PU + state * D + feature, numerator, feature < D)


@triton.jit
def split_kv_merge_kernel(
    PM,
    PL,
    PU,
    OUT,
    LSE,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    split, feature = tl.arange(0, BS), tl.arange(0, BD)
    state = row * SPLITS + split
    maxima = tl.load(PM + state, split < SPLITS, other=-float("inf"))
    lengths = tl.load(PL + state, split < SPLITS, other=0)
    active = lengths > 0
    maximum = tl.max(tl.where(active, maxima, -float("inf")), axis=0)
    # Replace inactive maxima BEFORE exp: tl.where evaluates both operands.
    safe_maxima = tl.where(active, maxima, maximum)
    correction = tl.where(active, tl.exp(safe_maxima - maximum), 0.0)
    denominator = tl.sum(lengths * correction, axis=0)
    numerators = tl.load(
        PU + state[:, None] * D + feature[None, :],
        (split[:, None] < SPLITS) & (feature[None, :] < D),
        other=0,
    )
    numerator = tl.sum(numerators * correction[:, None], axis=0)
    # LENGTHS>=1 ensures a finite global maximum and positive denominator.
    tl.store(OUT + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))
```

The second grid axis identifies the split. Scratch is private to each `(query,split)` pair; the merge has one writer per output query.

## 5. Python wrappers: scratch allocation and ordered launches

```python
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def split_decode_prevalidated(q, kc, vc, lengths, splits):
    batch, hq, d = q.shape
    _, hkv, capacity, _ = kc.shape
    maxima = torch.empty((batch, hq, splits), device=q.device)
    denominators = torch.empty_like(maxima)
    numerators = torch.empty((batch, hq, splits, d), device=q.device)
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    split_kv_partial_kernel[(batch*hq, splits)](q, kc, vc, lengths,
        maxima, denominators, numerators, hq, hkv, capacity, d, splits,
        triton.cdiv(capacity, splits), 32, triton.next_power_of_2(d), num_warps=4)
    split_kv_merge_kernel[(batch*hq,)](maxima, denominators, numerators, out, lse,
        d, splits, triton.next_power_of_2(splits), triton.next_power_of_2(d), num_warps=4)
    return out, lse


def split_decode_triton(q, kc, vc, lengths, splits=4):
    require_fp32(q, kc, vc)
    if q.ndim != 3 or kc.ndim != 4 or vc.shape != kc.shape:
        raise ValueError("require Q (B,Hq,D) and K/V (B,Hkv,C,D)")
    batch, hq, d = q.shape
    cb, hkv, capacity, cd = kc.shape
    if min(batch, hq, hkv, capacity) < 1 or batch != cb or d != cd or not 1 <= d <= 128 or hq % hkv:
        raise ValueError("require positive matching dimensions, D<=128, Hq%Hkv=0")
    if not isinstance(splits, int) or not 1 <= splits <= 128 or batch*hq*splits*d >= 2**31:
        raise ValueError("require S=1..128 and 32-bit-safe scratch offsets")
    if (lengths.dtype != torch.int32 or lengths.device != q.device
        or lengths.shape != (batch,) or not lengths.is_contiguous()):
        raise ValueError("lengths must be contiguous CUDA INT32 (B,)")
    if bool(((lengths < 1) | (lengths > capacity)).any()):
        raise ValueError("lengths must be in [1,C]")
    return split_decode_prevalidated(q, kc, vc, lengths, splits)


out, lse = split_decode_triton(q, kc, vc, lengths)
```

Length validation synchronizes with the GPU. Benchmark a prevalidated path after checking metadata and values once.

## 6. Compare one split, empty chunks, ragged lengths, and extreme scores

```python
expected, expected_lse = decode_torch(q, kc, vc, lengths)
for splits in (1, 2, 3, 8, 128):
    actual, actual_lse = split_decode_triton(q, kc, vc, lengths, splits)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-4, atol=2e-4)
# A single-token cache with S>C leaves all but the first chunk empty.
single_lengths = torch.ones(2, device=DEVICE, dtype=torch.int32)
single_k, single_v = kc[:, :, :1].contiguous(), vc[:, :, :1].contiguous()
actual, _ = split_decode_triton(q, single_k, single_v, single_lengths, 8)
torch.testing.assert_close(actual, single_v[:, :, 0].repeat_interleave(2, dim=1))
# Large finite score shifts should remain numerically stable.
sharp_q, sharp_k = (q * 10).contiguous(), (kc * 10).contiguous()
actual, actual_lse = split_decode_triton(sharp_q, sharp_k, vc, lengths, 8)
expected, expected_lse = decode_torch(sharp_q, sharp_k, vc, lengths)
torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
torch.testing.assert_close(actual_lse, expected_lse, rtol=3e-4, atol=3e-4)
print("Split merge, empty chunks, grouped heads, and stable-score checks passed.")
```

S=1 tests the unsplit limit. S=128 also tests non-power-of-two capacity and many completely empty chunks.

## 7. Benchmark the complete two-stage operation

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_q = torch.randn(1, 8, 64, device=DEVICE)
timed_k = torch.randn(1, 2, 2048, 64, device=DEVICE)
timed_v = torch.randn_like(timed_k)
timed_lengths = torch.tensor([2048], device=DEVICE, dtype=torch.int32)
reference, reference_lse = decode_torch(timed_q, timed_k, timed_v, timed_lengths)
for splits in (1, 2, 4, 8, 16):
    actual, actual_lse = split_decode_triton(timed_q, timed_k, timed_v, timed_lengths, splits)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)
    ms = bench_ms(lambda: split_decode_prevalidated(timed_q, timed_k, timed_v, timed_lengths, splits))
    print(f"S={splits:2}: partial + merge {ms*1000:.3f} us")
```

Each timing includes scratch/output allocation and both launches. S=1 still launches the merge kernel, so it is not identical to tutorial 22’s single-launch baseline. Small workloads can lose from extra launches and scratch traffic.

## 8. Account for scratch and plan a paged extension

```python
splits = 8
batch, hq, d = timed_q.shape
scratch_bytes = 4 * batch * hq * splits * (d + 2)
print("FP32 split state bytes:", scratch_bytes)
```

Scratch stores S maxima, S denominators, and S vectors per query. In a paged version, translate each logical token `t` through `TABLE[b,t//P]`; split intervals and softmax states stay unchanged. A reusable workspace can remove repeated allocation from a serving loop.

## Try next

Continue with [fused training operations](30_fused_training_operations.md). Tune split count for batch size, cache length, and device occupancy; retain the whole-operation correctness check as the schedule changes.

## 9. Complete code in one block

Run this block independently in a fresh notebook, or save it as a Python script. It combines sections 1–8; the kernel definitions match the linked Triton-only file.

```python
# 1. Imports and ragged grouped-query inputs
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


q = torch.randn(2, 4, 7, device=DEVICE)
kc = torch.randn(2, 2, 70, 7, device=DEVICE)
vc = torch.randn_like(kc)
lengths = torch.tensor([1, 65], device=DEVICE, dtype=torch.int32)


# 2. PyTorch eager: inspect one query over its prefix
scores = (kc[1, 0, :65] @ q[1, 0]) * 7**-0.5
print("One-head output:", (scores.softmax(0) @ vc[1, 0, :65]).cpu())
print("One-head LSE:", scores.logsumexp(0).item())


# 3. PyTorch reference: independent full-prefix attention
def decode_torch(q, kc, vc, lengths):
    batch, hq, d = q.shape
    hkv = kc.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    for b, length in enumerate(lengths.cpu().tolist()):
        for h in range(hq):
            kh = h // (hq // hkv)
            score = (kc[b, kh, :length] @ q[b, h]) * d**-0.5
            out[b, h] = score.softmax(0) @ vc[b, kh, :length]
            lse[b, h] = score.logsumexp(0)
    return out, lse


# 4. Triton: independent scans and a query-owned merge
@triton.jit
def split_kv_partial_kernel(
    Q,
    KC,
    VC,
    LENGTHS,
    PM,
    PL,
    PU,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, BT)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    begin = split * CHUNK
    end = tl.minimum(begin + CHUNK, length)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    # Empty chunks skip the loop, avoiding exp(-inf - -inf).
    for start in range(begin, end, BT):
        t = start + token
        offset = ((batch * HKV + kv_head) * C + t[:, None]) * D + feature[None, :]
        valid = (t[:, None] < end) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < end, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    state = row * SPLITS + split
    tl.store(PM + state, maximum)
    tl.store(PL + state, denominator)
    tl.store(PU + state * D + feature, numerator, feature < D)


@triton.jit
def split_kv_merge_kernel(
    PM,
    PL,
    PU,
    OUT,
    LSE,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    split, feature = tl.arange(0, BS), tl.arange(0, BD)
    state = row * SPLITS + split
    maxima = tl.load(PM + state, split < SPLITS, other=-float("inf"))
    lengths = tl.load(PL + state, split < SPLITS, other=0)
    active = lengths > 0
    maximum = tl.max(tl.where(active, maxima, -float("inf")), axis=0)
    # Replace inactive maxima BEFORE exp: tl.where evaluates both operands.
    safe_maxima = tl.where(active, maxima, maximum)
    correction = tl.where(active, tl.exp(safe_maxima - maximum), 0.0)
    denominator = tl.sum(lengths * correction, axis=0)
    numerators = tl.load(
        PU + state[:, None] * D + feature[None, :],
        (split[:, None] < SPLITS) & (feature[None, :] < D),
        other=0,
    )
    numerator = tl.sum(numerators * correction[:, None], axis=0)
    # LENGTHS>=1 ensures a finite global maximum and positive denominator.
    tl.store(OUT + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))


# 5. Python wrappers: scratch allocation and ordered launches
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def split_decode_prevalidated(q, kc, vc, lengths, splits):
    batch, hq, d = q.shape
    _, hkv, capacity, _ = kc.shape
    maxima = torch.empty((batch, hq, splits), device=q.device)
    denominators = torch.empty_like(maxima)
    numerators = torch.empty((batch, hq, splits, d), device=q.device)
    out = torch.empty_like(q)
    lse = torch.empty((batch, hq), device=q.device)
    split_kv_partial_kernel[(batch*hq, splits)](q, kc, vc, lengths,
        maxima, denominators, numerators, hq, hkv, capacity, d, splits,
        triton.cdiv(capacity, splits), 32, triton.next_power_of_2(d), num_warps=4)
    split_kv_merge_kernel[(batch*hq,)](maxima, denominators, numerators, out, lse,
        d, splits, triton.next_power_of_2(splits), triton.next_power_of_2(d), num_warps=4)
    return out, lse


def split_decode_triton(q, kc, vc, lengths, splits=4):
    require_fp32(q, kc, vc)
    if q.ndim != 3 or kc.ndim != 4 or vc.shape != kc.shape:
        raise ValueError("require Q (B,Hq,D) and K/V (B,Hkv,C,D)")
    batch, hq, d = q.shape
    cb, hkv, capacity, cd = kc.shape
    if min(batch, hq, hkv, capacity) < 1 or batch != cb or d != cd or not 1 <= d <= 128 or hq % hkv:
        raise ValueError("require positive matching dimensions, D<=128, Hq%Hkv=0")
    if not isinstance(splits, int) or not 1 <= splits <= 128 or batch*hq*splits*d >= 2**31:
        raise ValueError("require S=1..128 and 32-bit-safe scratch offsets")
    if (lengths.dtype != torch.int32 or lengths.device != q.device
        or lengths.shape != (batch,) or not lengths.is_contiguous()):
        raise ValueError("lengths must be contiguous CUDA INT32 (B,)")
    if bool(((lengths < 1) | (lengths > capacity)).any()):
        raise ValueError("lengths must be in [1,C]")
    return split_decode_prevalidated(q, kc, vc, lengths, splits)


out, lse = split_decode_triton(q, kc, vc, lengths)


# 6. Compare one split, empty chunks, ragged lengths, and extreme scores
expected, expected_lse = decode_torch(q, kc, vc, lengths)
for splits in (1, 2, 3, 8, 128):
    actual, actual_lse = split_decode_triton(q, kc, vc, lengths, splits)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-4, atol=2e-4)
# A single-token cache with S>C leaves all but the first chunk empty.
single_lengths = torch.ones(2, device=DEVICE, dtype=torch.int32)
single_k, single_v = kc[:, :, :1].contiguous(), vc[:, :, :1].contiguous()
actual, _ = split_decode_triton(q, single_k, single_v, single_lengths, 8)
torch.testing.assert_close(actual, single_v[:, :, 0].repeat_interleave(2, dim=1))
# Large finite score shifts should remain numerically stable.
sharp_q, sharp_k = (q * 10).contiguous(), (kc * 10).contiguous()
actual, actual_lse = split_decode_triton(sharp_q, sharp_k, vc, lengths, 8)
expected, expected_lse = decode_torch(sharp_q, sharp_k, vc, lengths)
torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
torch.testing.assert_close(actual_lse, expected_lse, rtol=3e-4, atol=3e-4)
print("Split merge, empty chunks, grouped heads, and stable-score checks passed.")


# 7. Benchmark the complete two-stage operation
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_q = torch.randn(1, 8, 64, device=DEVICE)
timed_k = torch.randn(1, 2, 2048, 64, device=DEVICE)
timed_v = torch.randn_like(timed_k)
timed_lengths = torch.tensor([2048], device=DEVICE, dtype=torch.int32)
reference, reference_lse = decode_torch(timed_q, timed_k, timed_v, timed_lengths)
for splits in (1, 2, 4, 8, 16):
    actual, actual_lse = split_decode_triton(timed_q, timed_k, timed_v, timed_lengths, splits)
    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(actual_lse, reference_lse, rtol=2e-4, atol=2e-4)
    ms = bench_ms(lambda: split_decode_prevalidated(timed_q, timed_k, timed_v, timed_lengths, splits))
    print(f"S={splits:2}: partial + merge {ms*1000:.3f} us")


# 8. Account for scratch and plan a paged extension
splits = 8
batch, hq, d = timed_q.shape
scratch_bytes = 4 * batch * hq * splits * (d + 2)
print("FP32 split state bytes:", scratch_bytes)
```
