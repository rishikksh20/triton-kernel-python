# 22. Decode attention: stream a single query over a ragged KV cache

Previous: [KV-cache updates](21_kv_cache_updates.md). Next: [Dropout](23_dropout.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Q is contiguous FP32 `(B,Hq,D)` and the independent K/V caches are contiguous FP32 `(B,Hkv,C,D)` on one NVIDIA CUDA GPU. D=1–128, Hq is a positive multiple of Hkv, and INT32 lengths `(B,)` lie in `[1,C]`. Inputs are finite moderate values. This inference-only kernel has no dropout and no registered backward operation.

## Concepts: one new query, many cached keys

At each decode step, a sequence contributes one query per query head. For batch b and head h, map to KV head g(h), and attend to its valid prefix:

$$
s_t=\frac{q_{b,h}\cdot K_{b,g(h),t}}{\sqrt D},\quad
p_t=\frac{e^{s_t}}{\sum_{j<L_b}e^{s_j}},\quad
o_{b,h}=\sum_{t<L_b}p_tV_{b,g(h),t}.
$$

### 1. The cache prefix supplies causality

The new K/V must already have been appended, and Q is the query for the latest token at position L_b-1. Every valid cached token is then allowed. A square triangular mask is unnecessary. This assumption does not apply to a chunk of multiple new queries or a query for an older position.

### 2. Grouped-query attention shares KV heads

With G=Hq/Hkv, use `g(h)=floor(h/G)`. Each group of G neighboring query heads shares one KV head. G=1 is ordinary multi-head attention; Hkv=1 is multi-query attention. This kernel handles all three with the same pointer mapping.

### 3. Reuse the online-softmax recurrence

For each key block, compute a new maximum m', correction alpha, unnormalized block weights w, denominator l', and vector numerator u':

$$
m'=\max(m,\max_t s_t),\quad \alpha=e^{m-m'},\quad
w_t=e^{s_t-m'},\quad
\ell'=\alpha\ell+\sum_t w_t,\quad
u'=\alpha u+\sum_t w_tV_t.
$$

Initialize m=-infinity, l=0, u=0, and return u/l. Padded keys have score -infinity and weight zero. Every sequence has at least one valid key, avoiding an empty softmax. See [online softmax](13_online_softmax.md) for the derivation.

```text
one Q row -> cache block -> scores -> online (max, denominator, numerator)
                         repeat over valid length
                         -> output vector and log-sum-exp
```

### 4. Decode needs a different parallel schedule from prefill

There is only one query row per head, so this guide uses vector reductions rather than a many-query dot tile. It stores no full score/probability tensor. One program per batch/query head scans its prefix, giving clear ownership but limited parallelism for a small batch. Split-KV implementations divide the prefix across programs and merge their online states; paging and split-KV are later extensions. The [Transformers caching guide](https://huggingface.co/docs/transformers/main/cache_explanation) explains the surrounding generation loop.

## 1. Imports and a tiny grouped-query example

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

q = torch.tensor([[[1., 0.], [0., 1.]]], device=DEVICE)
k_cache = torch.tensor([[[[1., 0.], [0., 1.], [99., 99.]]]], device=DEVICE)
v_cache = torch.tensor([[[[10., 0.], [0., 20.], [999., 999.]]]], device=DEVICE)
lengths = torch.tensor([2], device=DEVICE, dtype=torch.int32)
```

Two query heads share one KV head. The third cache slot deliberately contains large unused values.

## 2. PyTorch eager: use only the initialized prefix

```python
valid_k = k_cache[0, 0, :2]
valid_v = v_cache[0, 0, :2]
scores = q[0] @ valid_k.T / q.shape[-1]**0.5
o_eager = (scores.softmax(dim=-1) @ valid_v)[None]
print("Decode output:\n", o_eager.cpu())
```

The invalid third slot must contribute neither scores nor weighted values.

## 3. PyTorch function: broadcast KV groups and mask ragged lengths

```python
def decode_torch(q, k_cache, v_cache, lengths, return_lse=False):
    group = q.shape[1] // k_cache.shape[1]
    k = k_cache.repeat_interleave(group, dim=1)
    v = v_cache.repeat_interleave(group, dim=1)
    valid = torch.arange(k.shape[2], device=q.device)[None, :] < lengths[:, None]
    # Zero unused entries before arithmetic, allowing even NaN-filled capacity.
    k = torch.where(valid[:, None, :, None], k, 0.)
    v = torch.where(valid[:, None, :, None], v, 0.)
    scores = (q[:, :, None, :] * k).sum(-1) * q.shape[-1]**-0.5
    scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
    out = (scores.softmax(-1)[..., None] * v).sum(-2)
    lse = scores.logsumexp(-1)
    return (out, lse) if return_lse else out
```

The dense reference expands KV groups and materializes scores. These allocations make it useful for correctness but do not represent an optimized decode library.

## 4. Triton: a single-query online reduction over cache blocks

```python
@triton.jit
def decode_kernel(Q, KC, VC, LENGTHS, O, LSE, HQ: tl.constexpr, HKV: tl.constexpr,
                  C: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    cache_row = batch * HKV + kv_head
    feature = tl.arange(0, BD)
    token = tl.arange(0, BT)
    query = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    for start in range(0, length, BT):
        t = start + token
        mask = (t[:, None] < length) & (feature[None, :] < D)
        offsets = (cache_row * C + t[:, None]) * D + feature[None, :]
        key = tl.load(KC + offsets, mask, other=0)
        value = tl.load(VC + offsets, mask, other=0)
        scores = tl.sum(key * query[None, :], axis=1) * (D ** -0.5)
        scores = tl.where(t < length, scores, -float("inf"))
        updated = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - updated)
        weights = tl.exp(scores - updated)
        numerator = numerator * correction + tl.sum(weights[:, None] * value, axis=0)
        denominator = denominator * correction + tl.sum(weights, axis=0)
        maximum = updated
    tl.store(O + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))
```

The loop bound is each sequence’s runtime length. Invalid cache slots are never loaded; padded features are zero. Numerator and denominator both receive the maximum correction.

## 5. Python wrapper and a preallocated launch helper

```python
def decode_into(q, k_cache, v_cache, lengths, out, lse):
    b, hq, d = q.shape
    _, hkv, capacity, _ = k_cache.shape
    decode_kernel[(b*hq,)](q, k_cache, v_cache, lengths, out, lse,
                           hq, hkv, capacity, d, 32, triton.next_power_of_2(d), num_warps=4)


def decode_triton(q, k_cache, v_cache, lengths, return_lse=False):
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("require Q (B,Hq,D), matching K/V (B,Hkv,C,D)")
    b, hq, d = q.shape
    bk, hkv, capacity, dk = k_cache.shape
    if min(b, hq, hkv, capacity) < 1 or b != bk or d != dk or not 1 <= d <= 128 or hq % hkv:
        raise ValueError("incompatible batch/features, head grouping, or capacity")
    if lengths.shape != (b,) or lengths.dtype != torch.int32:
        raise ValueError("lengths must be an INT32 (B,) vector")
    values = (q, k_cache, v_cache)
    if any(x.dtype != torch.float32 for x in values):
        raise ValueError("Q/K/V must be FP32")
    if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (*values, lengths)):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if max(q.numel(), k_cache.numel()) >= 2**31:
        raise ValueError("require 32-bit-safe offsets")
    if lengths.min().item() < 1 or lengths.max().item() > capacity:
        raise ValueError("lengths must be in [1,capacity]")
    out = torch.empty_like(q)
    lse = torch.empty((b, hq), device=q.device, dtype=torch.float32)
    decode_into(q, k_cache, v_cache, lengths, out, lse)
    return (out, lse) if return_lse else out
```

The checked wrapper synchronizes to validate lengths. `decode_into` is a prevalidated launch path for timing and assumes the output/LSE buffers have matching shapes and dtype.

## 6. Compare outputs, log-sum-exp, and unused capacity

```python
torch.testing.assert_close(decode_triton(q, k_cache, v_cache, lengths), o_eager, rtol=1e-4, atol=1e-4)
for hq, hkv, d in ((1, 1, 1), (4, 4, 48), (4, 2, 64), (8, 1, 128)):
    query = torch.randn(3, hq, d, device=DEVICE)
    kc = torch.randn(3, hkv, 97, d, device=DEVICE)
    vc = torch.randn_like(kc)
    lens = torch.tensor([1, 33, 97], device=DEVICE, dtype=torch.int32)
    expected, expected_lse = decode_torch(query, kc, vc, lens, True)
    actual, lse = decode_triton(query, kc, vc, lens, True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-4)
    valid = torch.arange(97, device=DEVICE)[None, :] < lens[:, None]
    kc.masked_fill_(~valid[:, None, :, None], float("nan"))
    vc.masked_fill_(~valid[:, None, :, None], float("nan"))
    torch.testing.assert_close(decode_triton(query, kc, vc, lens), actual, rtol=0, atol=0)
print("MHA/GQA/MQA, ragged tails, LSE, and unread-capacity checks passed.")
```

NaN poisoning makes an accidental read of unused capacity visible. This differs from masking scores alone, which cannot prevent `0 * NaN` in the value product.

## 7. Benchmark preallocated decode at increasing cache lengths

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for length in (32, 128, 512, 1024):
    query = torch.randn(2, 8, 64, device=DEVICE)
    kc = torch.randn(2, 2, length, 64, device=DEVICE)
    vc = torch.randn_like(kc)
    lens = torch.full((2,), length, device=DEVICE, dtype=torch.int32)
    out, lse = decode_triton(query, kc, vc, lens, True)
    expected = decode_torch(query, kc, vc, lens)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    p = bench_ms(lambda: decode_torch(query, kc, vc, lens))
    t = bench_ms(lambda: decode_into(query, kc, vc, lens, out, lse))
    print(f"L={length:4d}: dense PyTorch {p*1000:.3f} us, preallocated Triton {t*1000:.3f} us")
```

Input creation, cache updates, length validation, compilation, and checks are outside timing. The dense reference allocates intermediates; Triton reuses output/LSE. This is an attention-only comparison with explicitly different allocation scopes.

## 8. Integrate append-before-read in a small generation loop

```python
kc = torch.zeros(1, 1, 4, 2, device=DEVICE)
vc = torch.zeros_like(kc)
lens = torch.zeros(1, device=DEVICE, dtype=torch.int32)
for position in range(4):
    new_k = torch.tensor([[[1., float(position)]]], device=DEVICE)
    new_v = new_k * 10
    # PyTorch append keeps this notebook independent of tutorial 21's definitions.
    kc[:, :, position] = new_k
    vc[:, :, position] = new_v
    lens += 1
    query = new_k.clone()
    actual = decode_triton(query, kc, vc, lens)
    expected = decode_torch(query, kc, vc, lens)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("Step", position, "output", actual.cpu().tolist())
```

Replace these indexed PyTorch writes with tutorial 21’s cache-update kernel after validating the standalone attention path. For a model using RoPE, Q and the new K must already carry their absolute-position rotations.

## Try next

Continue with [dropout](23_dropout.md). To improve small-batch decode parallelism, split the cache into chunks and merge `(m,l,u)` using a shared maximum and exponential corrections; compare both output and LSE.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a tiny grouped-query example
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

q = torch.tensor([[[1., 0.], [0., 1.]]], device=DEVICE)
k_cache = torch.tensor([[[[1., 0.], [0., 1.], [99., 99.]]]], device=DEVICE)
v_cache = torch.tensor([[[[10., 0.], [0., 20.], [999., 999.]]]], device=DEVICE)
lengths = torch.tensor([2], device=DEVICE, dtype=torch.int32)


# 2. PyTorch eager: use only the initialized prefix
valid_k = k_cache[0, 0, :2]
valid_v = v_cache[0, 0, :2]
scores = q[0] @ valid_k.T / q.shape[-1]**0.5
o_eager = (scores.softmax(dim=-1) @ valid_v)[None]
print("Decode output:\n", o_eager.cpu())


# 3. PyTorch function: broadcast KV groups and mask ragged lengths
def decode_torch(q, k_cache, v_cache, lengths, return_lse=False):
    group = q.shape[1] // k_cache.shape[1]
    k = k_cache.repeat_interleave(group, dim=1)
    v = v_cache.repeat_interleave(group, dim=1)
    valid = torch.arange(k.shape[2], device=q.device)[None, :] < lengths[:, None]
    # Zero unused entries before arithmetic, allowing even NaN-filled capacity.
    k = torch.where(valid[:, None, :, None], k, 0.)
    v = torch.where(valid[:, None, :, None], v, 0.)
    scores = (q[:, :, None, :] * k).sum(-1) * q.shape[-1]**-0.5
    scores = scores.masked_fill(~valid[:, None, :], -float("inf"))
    out = (scores.softmax(-1)[..., None] * v).sum(-2)
    lse = scores.logsumexp(-1)
    return (out, lse) if return_lse else out


# 4. Triton: a single-query online reduction over cache blocks
@triton.jit
def decode_kernel(Q, KC, VC, LENGTHS, O, LSE, HQ: tl.constexpr, HKV: tl.constexpr,
                  C: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    cache_row = batch * HKV + kv_head
    feature = tl.arange(0, BD)
    token = tl.arange(0, BT)
    query = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    for start in range(0, length, BT):
        t = start + token
        mask = (t[:, None] < length) & (feature[None, :] < D)
        offsets = (cache_row * C + t[:, None]) * D + feature[None, :]
        key = tl.load(KC + offsets, mask, other=0)
        value = tl.load(VC + offsets, mask, other=0)
        scores = tl.sum(key * query[None, :], axis=1) * (D ** -0.5)
        scores = tl.where(t < length, scores, -float("inf"))
        updated = tl.maximum(maximum, tl.max(scores, axis=0))
        correction = tl.exp(maximum - updated)
        weights = tl.exp(scores - updated)
        numerator = numerator * correction + tl.sum(weights[:, None] * value, axis=0)
        denominator = denominator * correction + tl.sum(weights, axis=0)
        maximum = updated
    tl.store(O + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))


# 5. Python wrapper and a preallocated launch helper
def decode_into(q, k_cache, v_cache, lengths, out, lse):
    b, hq, d = q.shape
    _, hkv, capacity, _ = k_cache.shape
    decode_kernel[(b*hq,)](q, k_cache, v_cache, lengths, out, lse,
                           hq, hkv, capacity, d, 32, triton.next_power_of_2(d), num_warps=4)


def decode_triton(q, k_cache, v_cache, lengths, return_lse=False):
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("require Q (B,Hq,D), matching K/V (B,Hkv,C,D)")
    b, hq, d = q.shape
    bk, hkv, capacity, dk = k_cache.shape
    if min(b, hq, hkv, capacity) < 1 or b != bk or d != dk or not 1 <= d <= 128 or hq % hkv:
        raise ValueError("incompatible batch/features, head grouping, or capacity")
    if lengths.shape != (b,) or lengths.dtype != torch.int32:
        raise ValueError("lengths must be an INT32 (B,) vector")
    values = (q, k_cache, v_cache)
    if any(x.dtype != torch.float32 for x in values):
        raise ValueError("Q/K/V must be FP32")
    if not q.is_cuda or any(x.device != q.device or not x.is_contiguous() for x in (*values, lengths)):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if max(q.numel(), k_cache.numel()) >= 2**31:
        raise ValueError("require 32-bit-safe offsets")
    if lengths.min().item() < 1 or lengths.max().item() > capacity:
        raise ValueError("lengths must be in [1,capacity]")
    out = torch.empty_like(q)
    lse = torch.empty((b, hq), device=q.device, dtype=torch.float32)
    decode_into(q, k_cache, v_cache, lengths, out, lse)
    return (out, lse) if return_lse else out


# 6. Compare outputs, log-sum-exp, and unused capacity
torch.testing.assert_close(decode_triton(q, k_cache, v_cache, lengths), o_eager, rtol=1e-4, atol=1e-4)
for hq, hkv, d in ((1, 1, 1), (4, 4, 48), (4, 2, 64), (8, 1, 128)):
    query = torch.randn(3, hq, d, device=DEVICE)
    kc = torch.randn(3, hkv, 97, d, device=DEVICE)
    vc = torch.randn_like(kc)
    lens = torch.tensor([1, 33, 97], device=DEVICE, dtype=torch.int32)
    expected, expected_lse = decode_torch(query, kc, vc, lens, True)
    actual, lse = decode_triton(query, kc, vc, lens, True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, expected_lse, rtol=1e-4, atol=1e-4)
    valid = torch.arange(97, device=DEVICE)[None, :] < lens[:, None]
    kc.masked_fill_(~valid[:, None, :, None], float("nan"))
    vc.masked_fill_(~valid[:, None, :, None], float("nan"))
    torch.testing.assert_close(decode_triton(query, kc, vc, lens), actual, rtol=0, atol=0)
print("MHA/GQA/MQA, ragged tails, LSE, and unread-capacity checks passed.")


# 7. Benchmark preallocated decode at increasing cache lengths
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for length in (32, 128, 512, 1024):
    query = torch.randn(2, 8, 64, device=DEVICE)
    kc = torch.randn(2, 2, length, 64, device=DEVICE)
    vc = torch.randn_like(kc)
    lens = torch.full((2,), length, device=DEVICE, dtype=torch.int32)
    out, lse = decode_triton(query, kc, vc, lens, True)
    expected = decode_torch(query, kc, vc, lens)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    p = bench_ms(lambda: decode_torch(query, kc, vc, lens))
    t = bench_ms(lambda: decode_into(query, kc, vc, lens, out, lse))
    print(f"L={length:4d}: dense PyTorch {p*1000:.3f} us, preallocated Triton {t*1000:.3f} us")


# 8. Integrate append-before-read in a small generation loop
kc = torch.zeros(1, 1, 4, 2, device=DEVICE)
vc = torch.zeros_like(kc)
lens = torch.zeros(1, device=DEVICE, dtype=torch.int32)
for position in range(4):
    new_k = torch.tensor([[[1., float(position)]]], device=DEVICE)
    new_v = new_k * 10
    # PyTorch append keeps this notebook independent of tutorial 21's definitions.
    kc[:, :, position] = new_k
    vc[:, :, position] = new_v
    lens += 1
    query = new_k.clone()
    actual = decode_triton(query, kc, vc, lens)
    expected = decode_torch(query, kc, vc, lens)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("Step", position, "output", actual.cpu().tolist())
```
