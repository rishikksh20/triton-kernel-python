# 21. KV-cache updates: append one token into preallocated storage

Previous: [RoPE](20_rope.md). Next: [Decode attention](22_decode_attention.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. K/V caches are independent contiguous FP32 tensors `(B,H,C,D)` on one NVIDIA CUDA GPU. New K/V have shape `(B,H,D)`, D=1–128, and each batch item supplies an INT32 append position in `[0,C)`. Prefixes before those positions are assumed initialized. The operation mutates both caches and returns new lengths; it has no registered backward operation.

## Concepts: append without copying the prefix

A static cache allocates space for C tokens once. At decode step b, position p_b is the number of tokens already cached for that sequence:

$$
K^{\rm cache}_{b,h,p_b,d}\leftarrow K^{\rm new}_{b,h,d},\quad
V^{\rm cache}_{b,h,p_b,d}\leftarrow V^{\rm new}_{b,h,d},\quad
L_b\leftarrow p_b+1.
$$

### 1. Capacity and logical length are different

Only the first L_b cache positions are valid. Unwritten capacity must not influence attention. Appending a single token should write O(BHD) values, independent of the existing prefix length. Repeated concatenation reallocates storage and copies prefixes instead.

The [Transformers cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation) describes reuse of past keys/values during autoregressive generation. This guide implements a static contiguous layout for one layer; it does not implement cache eviction or paging.

### 2. One program owns a batch/head slice

Flatten `(B,H)` into a program ID. Load the append position for that batch item, then use

$$
\text{offset}=((bH+h)C+p_b)D+d.
$$

Different batch/head pairs own disjoint cache memory even if their position numbers match. Head zero writes the new length, so each length has one writer. No atomics are necessary.

```text
new K/V (B,H,D) + old lengths/positions (B)
                -> indexed writes into (B,H,C,D)
                -> new lengths = positions + 1
                -> decode reads the valid prefixes
```

### 3. The stream orders the update before decode

Run the cache update before any kernel that reads the appended slot. Launches on the same CUDA stream preserve that order; other streams require explicit coordination. Cache tensors and new inputs must not alias. If using RoPE, rotate new K at the absolute token position before appending it. The stored K is already rotated and should not be rotated again.

Positions are append lengths, not arbitrary scatter indices: overwriting old valid entries violates the append interpretation. The wrapper checks bounds with a GPU synchronization. The benchmark uses already validated arguments and preallocated output lengths.

## 1. Imports and caches with unequal prefix lengths

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

b, h, capacity, d = 2, 2, 5, 3
k_cache = torch.full((b, h, capacity, d), -99., device=DEVICE)
v_cache = torch.full_like(k_cache, -77.)
positions = torch.tensor([0, 3], device=DEVICE, dtype=torch.int32)
# In a real decode loop these prefixes came from prefill or prior appends.
k_cache[1, :, :3] = 1.
v_cache[1, :, :3] = 2.
new_k = torch.arange(b*h*d, device=DEVICE).float().reshape(b, h, d)
new_v = new_k + 100.
```

The first sequence is empty; the second already has three valid entries.

## 2. PyTorch eager: append into each sequence

```python
eager_k, eager_v = k_cache.clone(), v_cache.clone()
for batch, position in enumerate(positions.cpu().tolist()):
    eager_k[batch, :, position, :] = new_k[batch]
    eager_v[batch, :, position, :] = new_v[batch]
print("New lengths:", (positions + 1).cpu())
print("Batch 1 appended key:\n", eager_k[1, :, 3].cpu())
```

The CPU loop is an explanatory reference and is not the performance baseline.

## 3. PyTorch function: vectorized indexed writes

```python
def append_torch(k_cache, v_cache, new_k, new_v, positions):
    batch = torch.arange(k_cache.shape[0], device=k_cache.device)
    k_cache[batch, :, positions.long(), :] = new_k
    v_cache[batch, :, positions.long(), :] = new_v
    return positions + 1
```

This function mutates both caches. Each new length includes the appended token.

## 4. Triton: one owner for each head and each length

```python
@triton.jit
def append_kernel(KC, VC, NK, NV, POS, LENGTHS, H: tl.constexpr,
                  C: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, head = row // H, row % H
    feature = tl.arange(0, BLOCK)
    position = tl.load(POS + batch)
    offset = (row * C + position) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, feature < D)
    tl.store(VC + offset, v, feature < D)
    if head == 0:
        tl.store(LENGTHS + batch, position + 1)
```

All stores touch the requested slot only. A padded feature lane has no side effect.

## 5. Python wrapper and a validated launch helper

```python
def append_into(k_cache, v_cache, new_k, new_v, positions, lengths):
    b, h, capacity, d = k_cache.shape
    append_kernel[(b*h,)](k_cache, v_cache, new_k, new_v, positions, lengths,
                          h, capacity, d, triton.next_power_of_2(d), num_warps=4)


def append_triton(k_cache, v_cache, new_k, new_v, positions):
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape or min(k_cache.shape) < 1:
        raise ValueError("require nonempty matching (B,H,C,D) caches")
    b, h, capacity, d = k_cache.shape
    if not 1 <= d <= 128 or k_cache.numel() >= 2**31:
        raise ValueError("require D=1..128 and 32-bit-safe offsets")
    if new_k.shape != (b, h, d) or new_v.shape != new_k.shape:
        raise ValueError("new K/V must have shape (B,H,D)")
    if positions.shape != (b,) or positions.dtype != torch.int32:
        raise ValueError("positions must be an INT32 (B,) vector")
    values = (k_cache, v_cache, new_k, new_v)
    if any(t.dtype != torch.float32 for t in values):
        raise ValueError("K/V must be FP32")
    if not k_cache.is_cuda or any(t.device != k_cache.device or not t.is_contiguous() for t in (*values, positions)):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if positions.min().item() < 0 or positions.max().item() >= capacity:
        raise ValueError("append position is outside cache capacity")
    lengths = torch.empty_like(positions)
    append_into(k_cache, v_cache, new_k, new_v, positions, lengths)
    return lengths
```

The wrapper checks shapes and positions; independent, non-overlapping K/V storage is an input contract. `append_into` reuses a separate INT32 length buffer and assumes prior validation.

## 6. Compare updates and verify untouched storage

```python
actual_k, actual_v = k_cache.clone(), v_cache.clone()
lengths = append_triton(actual_k, actual_v, new_k, new_v, positions)
torch.testing.assert_close(actual_k, eager_k, rtol=0, atol=0)
torch.testing.assert_close(actual_v, eager_v, rtol=0, atol=0)
torch.testing.assert_close(lengths, positions + 1, rtol=0, atol=0)
for d in (1, 3, 48, 128):
    kc = torch.randn(3, 2, 7, d, device=DEVICE)
    vc = torch.randn_like(kc)
    nk, nv = torch.randn(3, 2, d, device=DEVICE), torch.randn(3, 2, d, device=DEVICE)
    pos = torch.tensor([0, 3, 6], device=DEVICE, dtype=torch.int32)
    expected_k, expected_v = kc.clone(), vc.clone()
    expected_lengths = append_torch(expected_k, expected_v, nk, nv, pos)
    actual_lengths = append_triton(kc, vc, nk, nv, pos)
    torch.testing.assert_close(kc, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
print("Ragged positions, full-capacity boundary, and untouched-cache checks passed.")
```

Comparing the entire caches catches an accidental prefix overwrite or a write into an unused slot.

## 7. Benchmark preallocated repeated writes to fixed slots

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

kc = torch.zeros(8, 8, 512, 64, device=DEVICE)
vc = torch.zeros_like(kc)
nk, nv = torch.randn(8, 8, 64, device=DEVICE), torch.randn(8, 8, 64, device=DEVICE)
pos = torch.full((8,), 127, device=DEVICE, dtype=torch.int32)
out_lengths = append_triton(kc, vc, nk, nv, pos)
p = bench_ms(lambda: append_torch(kc, vc, nk, nv, pos))
t = bench_ms(lambda: append_into(kc, vc, nk, nv, pos, out_lengths))
print(f"PyTorch indexed updates: {p*1000:.3f} us; Triton preallocated update: {t*1000:.3f} us")
```

Each benchmark invocation overwrites the same valid slots with the same values; it does not advance a decode loop. Position validation, cache allocation, compilation, and checks are outside timing. PyTorch allocates index/length tensors, while Triton reuses its length buffer.

## 8. Advance several decode steps and reject a full cache

```python
kc = torch.zeros(1, 2, 3, 5, device=DEVICE)
vc = torch.zeros_like(kc)
pos = torch.zeros(1, device=DEVICE, dtype=torch.int32)
for step in range(3):
    nk = torch.full((1, 2, 5), float(step + 1), device=DEVICE)
    nv = nk * 10
    pos = append_triton(kc, vc, nk, nv, pos)
    assert pos.item() == step + 1
torch.testing.assert_close(kc[0, 0, :, 0], torch.tensor([1., 2., 3.], device=DEVICE))
try:
    append_triton(kc, vc, nk, nv, pos)
except ValueError:
    print("A full cache correctly rejects the next append.")
else:
    raise AssertionError("expected capacity validation")
```

Feed the returned lengths into the next append. Retaining the old positions would overwrite the same slot.

## Try next

Continue with [decode attention](22_decode_attention.md). Try a paged layout by replacing the contiguous position offset with a block-table lookup; preserve ownership, capacity checks, and append-before-read ordering.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and caches with unequal prefix lengths
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

b, h, capacity, d = 2, 2, 5, 3
k_cache = torch.full((b, h, capacity, d), -99., device=DEVICE)
v_cache = torch.full_like(k_cache, -77.)
positions = torch.tensor([0, 3], device=DEVICE, dtype=torch.int32)
# In a real decode loop these prefixes came from prefill or prior appends.
k_cache[1, :, :3] = 1.
v_cache[1, :, :3] = 2.
new_k = torch.arange(b*h*d, device=DEVICE).float().reshape(b, h, d)
new_v = new_k + 100.


# 2. PyTorch eager: append into each sequence
eager_k, eager_v = k_cache.clone(), v_cache.clone()
for batch, position in enumerate(positions.cpu().tolist()):
    eager_k[batch, :, position, :] = new_k[batch]
    eager_v[batch, :, position, :] = new_v[batch]
print("New lengths:", (positions + 1).cpu())
print("Batch 1 appended key:\n", eager_k[1, :, 3].cpu())


# 3. PyTorch function: vectorized indexed writes
def append_torch(k_cache, v_cache, new_k, new_v, positions):
    batch = torch.arange(k_cache.shape[0], device=k_cache.device)
    k_cache[batch, :, positions.long(), :] = new_k
    v_cache[batch, :, positions.long(), :] = new_v
    return positions + 1


# 4. Triton: one owner for each head and each length
@triton.jit
def append_kernel(KC, VC, NK, NV, POS, LENGTHS, H: tl.constexpr,
                  C: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    batch, head = row // H, row % H
    feature = tl.arange(0, BLOCK)
    position = tl.load(POS + batch)
    offset = (row * C + position) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, feature < D)
    tl.store(VC + offset, v, feature < D)
    if head == 0:
        tl.store(LENGTHS + batch, position + 1)


# 5. Python wrapper and a validated launch helper
def append_into(k_cache, v_cache, new_k, new_v, positions, lengths):
    b, h, capacity, d = k_cache.shape
    append_kernel[(b*h,)](k_cache, v_cache, new_k, new_v, positions, lengths,
                          h, capacity, d, triton.next_power_of_2(d), num_warps=4)


def append_triton(k_cache, v_cache, new_k, new_v, positions):
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape or min(k_cache.shape) < 1:
        raise ValueError("require nonempty matching (B,H,C,D) caches")
    b, h, capacity, d = k_cache.shape
    if not 1 <= d <= 128 or k_cache.numel() >= 2**31:
        raise ValueError("require D=1..128 and 32-bit-safe offsets")
    if new_k.shape != (b, h, d) or new_v.shape != new_k.shape:
        raise ValueError("new K/V must have shape (B,H,D)")
    if positions.shape != (b,) or positions.dtype != torch.int32:
        raise ValueError("positions must be an INT32 (B,) vector")
    values = (k_cache, v_cache, new_k, new_v)
    if any(t.dtype != torch.float32 for t in values):
        raise ValueError("K/V must be FP32")
    if not k_cache.is_cuda or any(t.device != k_cache.device or not t.is_contiguous() for t in (*values, positions)):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if positions.min().item() < 0 or positions.max().item() >= capacity:
        raise ValueError("append position is outside cache capacity")
    lengths = torch.empty_like(positions)
    append_into(k_cache, v_cache, new_k, new_v, positions, lengths)
    return lengths


# 6. Compare updates and verify untouched storage
actual_k, actual_v = k_cache.clone(), v_cache.clone()
lengths = append_triton(actual_k, actual_v, new_k, new_v, positions)
torch.testing.assert_close(actual_k, eager_k, rtol=0, atol=0)
torch.testing.assert_close(actual_v, eager_v, rtol=0, atol=0)
torch.testing.assert_close(lengths, positions + 1, rtol=0, atol=0)
for d in (1, 3, 48, 128):
    kc = torch.randn(3, 2, 7, d, device=DEVICE)
    vc = torch.randn_like(kc)
    nk, nv = torch.randn(3, 2, d, device=DEVICE), torch.randn(3, 2, d, device=DEVICE)
    pos = torch.tensor([0, 3, 6], device=DEVICE, dtype=torch.int32)
    expected_k, expected_v = kc.clone(), vc.clone()
    expected_lengths = append_torch(expected_k, expected_v, nk, nv, pos)
    actual_lengths = append_triton(kc, vc, nk, nv, pos)
    torch.testing.assert_close(kc, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(actual_lengths, expected_lengths, rtol=0, atol=0)
print("Ragged positions, full-capacity boundary, and untouched-cache checks passed.")


# 7. Benchmark preallocated repeated writes to fixed slots
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

kc = torch.zeros(8, 8, 512, 64, device=DEVICE)
vc = torch.zeros_like(kc)
nk, nv = torch.randn(8, 8, 64, device=DEVICE), torch.randn(8, 8, 64, device=DEVICE)
pos = torch.full((8,), 127, device=DEVICE, dtype=torch.int32)
out_lengths = append_triton(kc, vc, nk, nv, pos)
p = bench_ms(lambda: append_torch(kc, vc, nk, nv, pos))
t = bench_ms(lambda: append_into(kc, vc, nk, nv, pos, out_lengths))
print(f"PyTorch indexed updates: {p*1000:.3f} us; Triton preallocated update: {t*1000:.3f} us")


# 8. Advance several decode steps and reject a full cache
kc = torch.zeros(1, 2, 3, 5, device=DEVICE)
vc = torch.zeros_like(kc)
pos = torch.zeros(1, device=DEVICE, dtype=torch.int32)
for step in range(3):
    nk = torch.full((1, 2, 5), float(step + 1), device=DEVICE)
    nv = nk * 10
    pos = append_triton(kc, vc, nk, nv, pos)
    assert pos.item() == step + 1
torch.testing.assert_close(kc[0, 0, :, 0], torch.tensor([1., 2., 3.], device=DEVICE))
try:
    append_triton(kc, vc, nk, nv, pos)
except ValueError:
    print("A full cache correctly rejects the next append.")
else:
    raise AssertionError("expected capacity validation")
```
