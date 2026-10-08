# 20. RoPE: rotate adjacent feature pairs using absolute positions

Previous: [Quantized GEMM / FP8](19_quantized_gemm_fp8.md). Next: [KV-cache updates](21_kv_cache_updates.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 tensors `(B,H,T,D)` on one NVIDIA CUDA GPU, with even D=2–128 and finite moderate values. Positions are contiguous INT32 tensors `(B,T)`. Tables hold FP32 cosine/sine values for every supported position. The kernel computes forward rotation only and has no registered PyTorch backward operation.

## Concepts: turn positions into rotations

For pair j and absolute position p, define

$$
\omega_j=\theta^{-2j/D},\quad \phi_{p,j}=p\omega_j,\quad
\begin{pmatrix}y_{2j}\\y_{2j+1}\end{pmatrix}=
\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}
\begin{pmatrix}x_{2j}\\x_{2j+1}\end{pmatrix}.
$$

Here theta is 10000 and every feature participates in an **adjacent-pair** rotation. Other implementations pair the first and second halves of a vector; the table and feature convention must match the model. The [RoFormer paper](https://arxiv.org/abs/2104.09864) introduces rotary position embedding.

### 1. Rotate Q and K before computing attention

Apply the same position convention to Q and K, with their respective positions. A pair's dot product after both rotations depends on their relative angular displacement. This guide rotates a generic tensor; it can be called for both Q and K. V is ordinarily left unrotated.

### 2. A cache stores the already rotated key

During decode, a new token at absolute position p must use row p of the tables, even if its input tensor contains only one token. Using the local index zero at every decode step loses positional information. Store its rotated K once and read it unchanged on later steps.

```text
x (B,H,T,D) + positions (B,T) + tables (P,D/2)
        -> one program per batch/head/token
        -> load even/odd features and table row p
        -> rotate both members -> store output
```

### 3. Precomputed tables keep the kernel focused on memory access

Table generation uses PyTorch outside timing. The Triton kernel only gathers table entries and performs multiplies/adds, so it does not introduce a second trigonometric approximation. Position values must satisfy `0 <= p < P`; table bounds and D are part of the operation's contract. The wrapper checks position values, which synchronizes the GPU. Benchmarks therefore time validated, preallocated launches separately.

A rotation preserves each pair's squared norm in real arithmetic. Checking this invariant and the sign of a quarter-turn helps catch swapped coordinates or an incorrect sine sign.

## 1. Imports and a small positional example

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

x = torch.tensor([[[[1., 0., 0., 1.], [1., 0., 0., 1.]]]], device=DEVICE)
positions = torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)
```

A tensor has two adjacent pairs. Position zero should preserve all four features.

## 2. PyTorch eager: construct and apply the rotation tables

```python
d = x.shape[-1]
inverse_frequency = 10000.0 ** (-torch.arange(0, d, 2, device=DEVICE).float() / d)
angles = torch.arange(16, device=DEVICE).float()[:, None] * inverse_frequency[None, :]
cosine, sine = angles.cos().contiguous(), angles.sin().contiguous()
c = cosine[positions.long()][:, None, :, :]
s = sine[positions.long()][:, None, :, :]
y_eager = torch.empty_like(x)
y_eager[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
y_eager[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
print("Rotated features:\n", y_eager.cpu())
```

Tables are indexed by absolute positions and broadcast across heads.

## 3. PyTorch functions: reusable tables and adjacent-pair RoPE

```python
def rope_tables(max_position, d, device=DEVICE, base=10000.0):
    frequency = base ** (-torch.arange(0, d, 2, device=device).float() / d)
    angles = torch.arange(max_position, device=device).float()[:, None] * frequency[None, :]
    return angles.cos().contiguous(), angles.sin().contiguous()


def rope_torch(x, positions, cosine, sine):
    c, s = cosine[positions.long()][:, None], sine[positions.long()][:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = even * c - odd * s, even * s + odd * c
    return out
```

This reference explicitly follows adjacent-pair ordering rather than a model-specific half-vector convention.

## 4. Triton: gather a position table row and rotate pairs

```python
@triton.jit
def rope_kernel(X, POS, COS, SIN, Y, H: tl.constexpr, T: tl.constexpr,
                D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = row % T
    batch = row // (H * T)
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    position = tl.load(POS + batch * T + token)
    table_offset = position * (D // 2) + pair
    c = tl.load(COS + table_offset, valid, other=0)
    s = tl.load(SIN + table_offset, valid, other=0)
    even = tl.load(X + row * D + 2 * pair, valid, other=0)
    odd = tl.load(X + row * D + 2 * pair + 1, valid, other=0)
    tl.store(Y + row * D + 2 * pair, even * c - odd * s, valid)
    tl.store(Y + row * D + 2 * pair + 1, even * s + odd * c, valid)
```

Pairs can be padded to a power of two even when D itself is not a power of two. All feature/table loads and stores use the pair mask.

## 5. Python wrapper and a preallocated launch helper

```python
def rope_into(x, positions, cosine, sine, out):
    b, h, t, d = x.shape
    rope_kernel[(b*h*t,)](x, positions, cosine, sine, out, h, t, d,
                          triton.next_power_of_2(d // 2), num_warps=4, enable_fp_fusion=False)


def rope_triton(x, positions, cosine, sine):
    if x.ndim != 4 or x.dtype != torch.float32 or min(x.shape) < 1:
        raise ValueError("require a nonempty FP32 (B,H,T,D) tensor")
    b, h, t, d = x.shape
    if d % 2 or not 2 <= d <= 128 or x.numel() >= 2**31:
        raise ValueError("require even D=2..128 and 32-bit-safe offsets")
    if positions.shape != (b, t) or positions.dtype != torch.int32:
        raise ValueError("positions must be INT32 (B,T)")
    if cosine.ndim != 2 or cosine.shape != sine.shape or cosine.shape[0] < 1 or cosine.shape[1] != d//2:
        raise ValueError("tables must have matching (P,D/2) shapes")
    if cosine.dtype != torch.float32 or sine.dtype != torch.float32:
        raise ValueError("tables must be FP32")
    tensors = (x, positions, cosine, sine)
    if not x.is_cuda or any(v.device != x.device or not v.is_contiguous() for v in tensors):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if positions.min().item() < 0 or positions.max().item() >= cosine.shape[0]:
        raise ValueError("positions must index existing table rows")
    out = torch.empty_like(x)
    rope_into(x, positions, cosine, sine, out)
    return out
```

`rope_into` assumes its arguments have already been validated and its output matches x. Disabling multiply-add fusion makes comparison with the two explicit reference expressions particularly easy to inspect.

## 6. Compare values, norms, and uneven feature counts

```python
torch.testing.assert_close(rope_triton(x, positions, cosine, sine), y_eager, rtol=1e-5, atol=1e-6)
for b, h, t, d in ((1, 1, 1, 2), (2, 3, 7, 6), (2, 4, 33, 48), (1, 2, 5, 128)):
    sample = torch.randn(b, h, t, d, device=DEVICE)
    pos = torch.randint(0, 64, (b, t), device=DEVICE, dtype=torch.int32)
    cos, sin = rope_tables(64, d)
    actual, expected = rope_triton(sample, pos, cos, sin), rope_torch(sample, pos, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.square().sum(-1), sample.square().sum(-1), rtol=1e-5, atol=1e-5)
print("RoPE value, norm, position, and pair-tail checks passed.")
```

Random positions vary independently across batch items, which tests the mapping from the flattened program row back to `(batch,token)`.

## 7. Benchmark validated preallocated rotation

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(2, 8, 128, 64, device=DEVICE)
pos = torch.arange(128, device=DEVICE, dtype=torch.int32)[None].expand(2, -1).contiguous()
cos, sin = rope_tables(256, 64)
out = rope_triton(sample, pos, cos, sin)
p = bench_ms(lambda: rope_torch(sample, pos, cos, sin))
t = bench_ms(lambda: rope_into(sample, pos, cos, sin, out))
print(f"PyTorch allocating reference: {p*1000:.3f} us; Triton preallocated launch: {t*1000:.3f} us")
```

The table construction and synchronized position validation are outside timing. The PyTorch reference allocates intermediates/output while the Triton launch reuses output storage. This measures those stated paths, not equal allocation scopes; no speedup is assumed.

## 8. Check a quarter-turn and a one-token decode offset

```python
unit = torch.tensor([[[[1., 0.]]]], device=DEVICE)
quarter_cos = torch.zeros(1, 1, device=DEVICE)
quarter_sin = torch.ones(1, 1, device=DEVICE)
zero_pos = torch.zeros(1, 1, device=DEVICE, dtype=torch.int32)
rotated = rope_triton(unit, zero_pos, quarter_cos, quarter_sin)
torch.testing.assert_close(rotated, torch.tensor([[[[0., 1.]]]], device=DEVICE), rtol=0, atol=0)
cos, sin = rope_tables(64, 2)
decode_pos = torch.tensor([[37]], device=DEVICE, dtype=torch.int32)
torch.testing.assert_close(rope_triton(unit, decode_pos, cos, sin),
                           rope_torch(unit, decode_pos, cos, sin), rtol=1e-5, atol=1e-6)
print("Quarter-turn sign and absolute decode position passed.")
```

A length-one input can refer to position 37. The sequence dimension and absolute position are different quantities.

## Try next

Continue with [KV-cache updates](21_kv_cache_updates.md). Extend to partial rotary dimensions or the half-vector convention only after writing an independent reference for the chosen layout.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a small positional example
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

x = torch.tensor([[[[1., 0., 0., 1.], [1., 0., 0., 1.]]]], device=DEVICE)
positions = torch.tensor([[0, 1]], device=DEVICE, dtype=torch.int32)


# 2. PyTorch eager: construct and apply the rotation tables
d = x.shape[-1]
inverse_frequency = 10000.0 ** (-torch.arange(0, d, 2, device=DEVICE).float() / d)
angles = torch.arange(16, device=DEVICE).float()[:, None] * inverse_frequency[None, :]
cosine, sine = angles.cos().contiguous(), angles.sin().contiguous()
c = cosine[positions.long()][:, None, :, :]
s = sine[positions.long()][:, None, :, :]
y_eager = torch.empty_like(x)
y_eager[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
y_eager[..., 1::2] = x[..., 0::2] * s + x[..., 1::2] * c
print("Rotated features:\n", y_eager.cpu())


# 3. PyTorch functions: reusable tables and adjacent-pair RoPE
def rope_tables(max_position, d, device=DEVICE, base=10000.0):
    frequency = base ** (-torch.arange(0, d, 2, device=device).float() / d)
    angles = torch.arange(max_position, device=device).float()[:, None] * frequency[None, :]
    return angles.cos().contiguous(), angles.sin().contiguous()


def rope_torch(x, positions, cosine, sine):
    c, s = cosine[positions.long()][:, None], sine[positions.long()][:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = even * c - odd * s, even * s + odd * c
    return out


# 4. Triton: gather a position table row and rotate pairs
@triton.jit
def rope_kernel(X, POS, COS, SIN, Y, H: tl.constexpr, T: tl.constexpr,
                D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    token = row % T
    batch = row // (H * T)
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    position = tl.load(POS + batch * T + token)
    table_offset = position * (D // 2) + pair
    c = tl.load(COS + table_offset, valid, other=0)
    s = tl.load(SIN + table_offset, valid, other=0)
    even = tl.load(X + row * D + 2 * pair, valid, other=0)
    odd = tl.load(X + row * D + 2 * pair + 1, valid, other=0)
    tl.store(Y + row * D + 2 * pair, even * c - odd * s, valid)
    tl.store(Y + row * D + 2 * pair + 1, even * s + odd * c, valid)


# 5. Python wrapper and a preallocated launch helper
def rope_into(x, positions, cosine, sine, out):
    b, h, t, d = x.shape
    rope_kernel[(b*h*t,)](x, positions, cosine, sine, out, h, t, d,
                          triton.next_power_of_2(d // 2), num_warps=4, enable_fp_fusion=False)


def rope_triton(x, positions, cosine, sine):
    if x.ndim != 4 or x.dtype != torch.float32 or min(x.shape) < 1:
        raise ValueError("require a nonempty FP32 (B,H,T,D) tensor")
    b, h, t, d = x.shape
    if d % 2 or not 2 <= d <= 128 or x.numel() >= 2**31:
        raise ValueError("require even D=2..128 and 32-bit-safe offsets")
    if positions.shape != (b, t) or positions.dtype != torch.int32:
        raise ValueError("positions must be INT32 (B,T)")
    if cosine.ndim != 2 or cosine.shape != sine.shape or cosine.shape[0] < 1 or cosine.shape[1] != d//2:
        raise ValueError("tables must have matching (P,D/2) shapes")
    if cosine.dtype != torch.float32 or sine.dtype != torch.float32:
        raise ValueError("tables must be FP32")
    tensors = (x, positions, cosine, sine)
    if not x.is_cuda or any(v.device != x.device or not v.is_contiguous() for v in tensors):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    if positions.min().item() < 0 or positions.max().item() >= cosine.shape[0]:
        raise ValueError("positions must index existing table rows")
    out = torch.empty_like(x)
    rope_into(x, positions, cosine, sine, out)
    return out


# 6. Compare values, norms, and uneven feature counts
torch.testing.assert_close(rope_triton(x, positions, cosine, sine), y_eager, rtol=1e-5, atol=1e-6)
for b, h, t, d in ((1, 1, 1, 2), (2, 3, 7, 6), (2, 4, 33, 48), (1, 2, 5, 128)):
    sample = torch.randn(b, h, t, d, device=DEVICE)
    pos = torch.randint(0, 64, (b, t), device=DEVICE, dtype=torch.int32)
    cos, sin = rope_tables(64, d)
    actual, expected = rope_triton(sample, pos, cos, sin), rope_torch(sample, pos, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.square().sum(-1), sample.square().sum(-1), rtol=1e-5, atol=1e-5)
print("RoPE value, norm, position, and pair-tail checks passed.")


# 7. Benchmark validated preallocated rotation
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(2, 8, 128, 64, device=DEVICE)
pos = torch.arange(128, device=DEVICE, dtype=torch.int32)[None].expand(2, -1).contiguous()
cos, sin = rope_tables(256, 64)
out = rope_triton(sample, pos, cos, sin)
p = bench_ms(lambda: rope_torch(sample, pos, cos, sin))
t = bench_ms(lambda: rope_into(sample, pos, cos, sin, out))
print(f"PyTorch allocating reference: {p*1000:.3f} us; Triton preallocated launch: {t*1000:.3f} us")


# 8. Check a quarter-turn and a one-token decode offset
unit = torch.tensor([[[[1., 0.]]]], device=DEVICE)
quarter_cos = torch.zeros(1, 1, device=DEVICE)
quarter_sin = torch.ones(1, 1, device=DEVICE)
zero_pos = torch.zeros(1, 1, device=DEVICE, dtype=torch.int32)
rotated = rope_triton(unit, zero_pos, quarter_cos, quarter_sin)
torch.testing.assert_close(rotated, torch.tensor([[[[0., 1.]]]], device=DEVICE), rtol=0, atol=0)
cos, sin = rope_tables(64, 2)
decode_pos = torch.tensor([[37]], device=DEVICE, dtype=torch.int32)
torch.testing.assert_close(rope_triton(unit, decode_pos, cos, sin),
                           rope_torch(unit, decode_pos, cos, sin), rtol=1e-5, atol=1e-6)
print("Quarter-turn sign and absolute decode position passed.")
```
