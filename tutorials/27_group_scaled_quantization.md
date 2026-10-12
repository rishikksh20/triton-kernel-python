# 27. Group-scaled quantization: apply each scale inside the GEMM reduction

Previous: [MoE routing](26_moe_routing.md). Next: [Paged KV caches](28_paged_kv_caches.md). Start of the series: [vector addition](01_vector_addition.md).

This self-contained Markdown notebook extends [row quantization](18_quantization_dequantization.md) and [scaled GEMM](19_quantized_gemm_fp8.md). Run sections 1–8 in order, or use section 9 independently. X is contiguous FP32 `(M,K)` and W is contiguous FP32 `(N,K)` on one NVIDIA CUDA GPU. We compute `X @ W.T`. Dimensions are positive; feature-group size G is 32, 64, or 128; all buffer element counts are below `2**31`. Values are finite and scales must remain representable in FP32. This example uses symmetric INT8, FP32 scales, and FP32 output, with no autograd registration.

The commented device code is in [group_scaled_quantization.py](../kernels/group_scaled_quantization.py). The PyTorch code here allocates buffers, checks the result, and measures timings.

## Concepts: a row can need more than one scale

A row may contain a small-valued region and a large outlier in a different region. One scale for the entire row forces both regions to use the outlier's spacing. Dividing the feature dimension into groups lets each region choose its own spacing.

For row r and group g, with features `g*G <= k < min((g+1)*G,K)`, define

$$
s_{r,g}=\begin{cases}\max_k |x_{r,k}|/127&\max_k|x_{r,k}|>0\\1&\text{otherwise},\end{cases}
\quad q_{r,k}=\operatorname{clip}(\operatorname{round}(x_{r,k}/s_{r,g}),-127,127),
\quad \hat x_{r,k}=q_{r,k}s_{r,g}.
$$

We explicitly use round-half-away-from-zero: `sign(z)*floor(abs(z)+0.5)`. A zero group gets scale 1 and encoded zeros. A partial final group includes only real features; padded values are zero. For example, K=70 with G=32 gives three groups of 32, 32, and 6 features. The scale tensor has shape `(rows,3)`.

### 1. The K reduction must respect group boundaries

Both X and W are quantized along K. Their approximated product is

$$
C_{m,n}=\sum_g s^X_{m,g}s^W_{n,g}
    \left(\sum_{k\in g}q^X_{m,k}q^W_{n,k}\right).
$$

Unlike tutorial 19's per-row scaling, the scale product changes inside the K reduction. Compute one group's INT32 dot, convert it to FP32, multiply by that group's scales, and then add it to the FP32 output accumulator. A single scale applied after summing all integer dots cannot represent different groups.

```text
K group 0 -> INT32 dot -> scale A[:,0]*B[:,0] --+
K group 1 -> INT32 dot -> scale A[:,1]*B[:,1] --+--> FP32 sum -> C
K group 2 -> INT32 dot -> scale A[:,2]*B[:,2] --+
```

### 2. Separate encoding error from kernel error

Compare Triton GEMM with `dequantized_X @ dequantized_W.T` first. That tests the implementation using exactly the represented values. Comparing with the original FP32 product measures quantization error as well. Do not demand bitwise agreement with the original product after a lossy encoding.

Each group sum is bounded by `G*127**2`, safely inside INT32 for these group sizes. The output sums scaled groups in FP32; it does not keep a long integer accumulator. See the [Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) for input and accumulator types.

## 1. Imports and rows with different feature magnitudes

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


g = 32
x = torch.cat((torch.ones(2, 32), torch.full((2, 6), 40.)), dim=1).to(DEVICE)
w = torch.randn(3, 38, device=DEVICE)
```

The short last group exercises K padding. Each row has a small region and a much larger region.

## 2. PyTorch eager: inspect one group scale

```python
first_scale = x[:, :g].abs().amax(dim=1) / 127
last_scale = x[:, g:].abs().amax(dim=1) / 127
print("First-group scales:", first_scale.cpu())
print("Last-group scales:", last_scale.cpu())
print("Original GEMM:\n", (x @ w.T).cpu())
```

The two regions use different scales; the original GEMM is the accuracy baseline.

## 3. PyTorch functions: encode and reconstruct independently

```python
def quantize_groups_torch(x, group_size):
    rows, k = x.shape
    groups = triton.cdiv(k, group_size)
    padded = F.pad(x, (0, groups * group_size - k)).reshape(rows, groups, group_size)
    maximum = padded.abs().amax(dim=-1)
    scales = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    z = padded / scales[..., None]
    q = (z.sign() * (z.abs() + 0.5).floor()).clamp(-127, 127).to(torch.int8)
    return q.reshape(rows, -1)[:, :k].contiguous(), scales


def dequantize_groups_torch(q, scales, group_size):
    k = q.shape[1]
    groups = torch.arange(k, device=q.device) // group_size
    return q.float() * scales[:, groups]
```

Padding contributes zeros to maxima and is removed before returning the encoded matrix.

## 4. Triton: quantize, dequantize, and scale every partial dot

```python
@triton.jit
def group_quantize_kernel(X, Q, S, K: tl.constexpr, G: tl.constexpr):
    # One program owns one row's feature group, including a possible K tail.
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row * K + k, k < K, other=0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
    # Explicit rounding keeps the reference independent of cast rounding modes.
    scaled = x / scale
    magnitude = tl.floor(tl.abs(scaled) + 0.5)
    rounded = tl.where(scaled < 0, -magnitude, magnitude)
    encoded = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
    tl.store(Q + row * K + k, encoded, k < K)
    tl.store(S + row * tl.cdiv(K, G) + group, scale)


@triton.jit
def group_dequantize_kernel(
    Q,
    S,
    XHAT,
    SIZE: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # A feature's scale depends on both its row and its group along K.
    row, feature = index // K, index % K
    q = tl.load(Q + index, index < SIZE, other=0).to(tl.float32)
    scale = tl.load(S + row * tl.cdiv(K, G) + feature // G, index < SIZE, other=0)
    tl.store(XHAT + index, q * scale, index < SIZE)


@triton.jit
def group_scaled_gemm_kernel(
    A,
    BT,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, G)
    groups: tl.constexpr = tl.cdiv(K, G)
    result = tl.zeros((BM, BN), tl.float32)
    for group in range(groups):
        k = group * G + kk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            BT + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        # Reset the integer accumulator for every group: scales change with K.
        partial = tl.dot(a, b, tl.zeros((BM, BN), tl.int32))
        sa = tl.load(SA + rows * groups + group, rows < M, other=0)
        sb = tl.load(SB + cols * groups + group, cols < N, other=0)
        result += partial.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )
```

G doubles as the dot-reduction tile width. This keeps group boundaries aligned with dot boundaries.

## 5. Python wrappers: allocate scale matrices and launch tiles

```python
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def quantize_groups_triton(x, group_size=32):
    require_fp32(x)
    if x.ndim != 2 or min(x.shape) < 1 or group_size not in (32, 64, 128):
        raise ValueError("require a nonempty matrix and G in {32,64,128}")
    rows, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((rows, triton.cdiv(k, group_size)), device=x.device)
    group_quantize_kernel[(rows, scales.shape[1])](x, q, scales, k, group_size, num_warps=4)
    return q, scales


def dequantize_groups_triton(q, scales, group_size=32):
    require_fp32(scales)
    if q.ndim != 2 or min(q.shape) < 1 or group_size not in (32, 64, 128):
        raise ValueError("require a nonempty encoded matrix and supported G")
    rows, k = q.shape
    if (q.dtype != torch.int8 or q.device != scales.device or not q.is_contiguous()
        or q.numel() >= 2**31 or scales.shape != (rows, triton.cdiv(k, group_size))):
        raise ValueError("require INT8 Q and matching group scales on one device")
    out = torch.empty_like(q, dtype=torch.float32)
    group_dequantize_kernel[(triton.cdiv(q.numel(), 256),)](
        q, scales, out, q.numel(), k, group_size, 256, num_warps=4)
    return out


def grouped_gemm_triton(a, bt, sa, sb, group_size=32):
    require_fp32(sa, sb)
    if (a.ndim != 2 or bt.ndim != 2 or a.shape[1] != bt.shape[1]
        or group_size not in (32, 64, 128)):
        raise ValueError("require A (M,K), BT (N,K), and supported G")
    m, k = a.shape
    n = bt.shape[0]
    if min(m, n, k) < 1 or max(a.numel(), bt.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions and 32-bit-safe offsets")
    if any(t.dtype != torch.int8 or t.device != sa.device or not t.is_contiguous() for t in (a, bt)):
        raise ValueError("A and BT must be contiguous INT8 on the scale device")
    groups = triton.cdiv(k, group_size)
    if sa.shape != (m, groups) or sb.shape != (n, groups):
        raise ValueError("scale matrices must have one value per row and K group")
    out = torch.empty((m, n), device=a.device)
    group_scaled_gemm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        a, bt, sa, sb, out, m, n, k, group_size, 32, 32, num_warps=4)
    return out


qx, sx = quantize_groups_triton(x, g)
qw, sw = quantize_groups_triton(w, g)
y = grouped_gemm_triton(qx, qw, sx, sw, g)
```

Quantize weights once when reusing them. Requantizing both operands on every call has a different cost from GEMM alone.

## 6. Compare zero groups, rounding ties, and M/N/K tails

```python
for m, n, k, group_size in ((1, 1, 1, 32), (17, 19, 70, 32), (33, 35, 129, 64), (5, 7, 131, 128)):
    a = torch.randn(m, k, device=DEVICE)
    b = torch.randn(n, k, device=DEVICE)
    a[0] = 0
    aq, sa = quantize_groups_triton(a, group_size)
    bq, sb = quantize_groups_triton(b, group_size)
    rq, rs = quantize_groups_torch(a, group_size)
    torch.testing.assert_close(aq, rq, rtol=0, atol=0)
    torch.testing.assert_close(sa, rs, rtol=1e-6, atol=1e-7)
    da = dequantize_groups_triton(aq, sa, group_size)
    db = dequantize_groups_torch(bq, sb, group_size)
    torch.testing.assert_close(da, dequantize_groups_torch(aq, sa, group_size))
    torch.testing.assert_close(grouped_gemm_triton(aq, bq, sa, sb, group_size),
                               da @ db.T, rtol=2e-4, atol=2e-4)
ties = torch.tensor([[127., 0.5, -0.5, 1.5, -1.5]], device=DEVICE)
encoded, _ = quantize_groups_triton(ties)
assert encoded.cpu().tolist() == [[127, 1, -1, 2, -2]]
print("Group scales, zero rows, rounding ties, and tile tails passed.")
```

The half-integer test fixes the rounding rule. The GEMM comparison uses reconstructed operands to isolate kernel arithmetic.

## 7. Benchmark prequantized GEMM

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


sample_x = torch.randn(128, 256, device=DEVICE)
sample_w = torch.randn(192, 256, device=DEVICE)
aq, sa = quantize_groups_triton(sample_x)
bq, sb = quantize_groups_triton(sample_w)
da, db = dequantize_groups_torch(aq, sa, 32), dequantize_groups_torch(bq, sb, 32)
torch.testing.assert_close(grouped_gemm_triton(aq, bq, sa, sb), da @ db.T, rtol=2e-4, atol=3e-4)
p = bench_ms(lambda: da @ db.T)
t = bench_ms(lambda: grouped_gemm_triton(aq, bq, sa, sb))
print(f"Reconstructed FP32 GEMM {p*1000:.3f} us; grouped INT8 GEMM {t*1000:.3f} us")
```

Encoding is outside both timings. Both paths allocate an output; the FP32 baseline has already reconstructed its operands. Measure encoding separately when studying end-to-end inference. No speedup is assumed.

## 8. Measure representation error separately

```python
approximate = grouped_gemm_triton(aq, bq, sa, sb)
exact = sample_x @ sample_w.T
relative_error = (approximate - exact).norm() / exact.norm().clamp_min(1e-12)
print("Relative error against original FP32 inputs:", relative_error.item())
```

Try several group sizes on the same inputs. Smaller groups add scale traffic and partial dots while giving each group a narrower value range. This baseline supports full, contiguous feature groups, rather than arbitrary scale layouts.

## Try next

Continue with [paged KV caches](28_paged_kv_caches.md). To extend this GEMM, try weight-only quantization or an FP8 group format while keeping the scale application inside the reduction.

## 9. Complete code in one block

Run this block independently in a fresh notebook, or save it as a Python script. It combines sections 1–8; the kernel definitions match the linked Triton-only file.

```python
# 1. Imports and rows with different feature magnitudes
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


g = 32
x = torch.cat((torch.ones(2, 32), torch.full((2, 6), 40.)), dim=1).to(DEVICE)
w = torch.randn(3, 38, device=DEVICE)


# 2. PyTorch eager: inspect one group scale
first_scale = x[:, :g].abs().amax(dim=1) / 127
last_scale = x[:, g:].abs().amax(dim=1) / 127
print("First-group scales:", first_scale.cpu())
print("Last-group scales:", last_scale.cpu())
print("Original GEMM:\n", (x @ w.T).cpu())


# 3. PyTorch functions: encode and reconstruct independently
def quantize_groups_torch(x, group_size):
    rows, k = x.shape
    groups = triton.cdiv(k, group_size)
    padded = F.pad(x, (0, groups * group_size - k)).reshape(rows, groups, group_size)
    maximum = padded.abs().amax(dim=-1)
    scales = torch.where(maximum > 0, maximum / 127, torch.ones_like(maximum))
    z = padded / scales[..., None]
    q = (z.sign() * (z.abs() + 0.5).floor()).clamp(-127, 127).to(torch.int8)
    return q.reshape(rows, -1)[:, :k].contiguous(), scales


def dequantize_groups_torch(q, scales, group_size):
    k = q.shape[1]
    groups = torch.arange(k, device=q.device) // group_size
    return q.float() * scales[:, groups]


# 4. Triton: quantize, dequantize, and scale every partial dot
@triton.jit
def group_quantize_kernel(X, Q, S, K: tl.constexpr, G: tl.constexpr):
    # One program owns one row's feature group, including a possible K tail.
    row, group = tl.program_id(0), tl.program_id(1)
    k = group * G + tl.arange(0, G)
    x = tl.load(X + row * K + k, k < K, other=0)
    maximum = tl.max(tl.abs(x), axis=0)
    scale = tl.where(maximum > 0, maximum / 127.0, 1.0)
    # Explicit rounding keeps the reference independent of cast rounding modes.
    scaled = x / scale
    magnitude = tl.floor(tl.abs(scaled) + 0.5)
    rounded = tl.where(scaled < 0, -magnitude, magnitude)
    encoded = tl.minimum(127.0, tl.maximum(-127.0, rounded)).to(tl.int8)
    tl.store(Q + row * K + k, encoded, k < K)
    tl.store(S + row * tl.cdiv(K, G) + group, scale)


@triton.jit
def group_dequantize_kernel(
    Q,
    S,
    XHAT,
    SIZE: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # A feature's scale depends on both its row and its group along K.
    row, feature = index // K, index % K
    q = tl.load(Q + index, index < SIZE, other=0).to(tl.float32)
    scale = tl.load(S + row * tl.cdiv(K, G) + feature // G, index < SIZE, other=0)
    tl.store(XHAT + index, q * scale, index < SIZE)


@triton.jit
def group_scaled_gemm_kernel(
    A,
    BT,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, G)
    groups: tl.constexpr = tl.cdiv(K, G)
    result = tl.zeros((BM, BN), tl.float32)
    for group in range(groups):
        k = group * G + kk
        a = tl.load(
            A + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            BT + cols[None, :] * K + k[:, None],
            (cols[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        # Reset the integer accumulator for every group: scales change with K.
        partial = tl.dot(a, b, tl.zeros((BM, BN), tl.int32))
        sa = tl.load(SA + rows * groups + group, rows < M, other=0)
        sb = tl.load(SB + cols * groups + group, cols < N, other=0)
        result += partial.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


# 5. Python wrappers: allocate scale matrices and launch tiles
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def quantize_groups_triton(x, group_size=32):
    require_fp32(x)
    if x.ndim != 2 or min(x.shape) < 1 or group_size not in (32, 64, 128):
        raise ValueError("require a nonempty matrix and G in {32,64,128}")
    rows, k = x.shape
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty((rows, triton.cdiv(k, group_size)), device=x.device)
    group_quantize_kernel[(rows, scales.shape[1])](x, q, scales, k, group_size, num_warps=4)
    return q, scales


def dequantize_groups_triton(q, scales, group_size=32):
    require_fp32(scales)
    if q.ndim != 2 or min(q.shape) < 1 or group_size not in (32, 64, 128):
        raise ValueError("require a nonempty encoded matrix and supported G")
    rows, k = q.shape
    if (q.dtype != torch.int8 or q.device != scales.device or not q.is_contiguous()
        or q.numel() >= 2**31 or scales.shape != (rows, triton.cdiv(k, group_size))):
        raise ValueError("require INT8 Q and matching group scales on one device")
    out = torch.empty_like(q, dtype=torch.float32)
    group_dequantize_kernel[(triton.cdiv(q.numel(), 256),)](
        q, scales, out, q.numel(), k, group_size, 256, num_warps=4)
    return out


def grouped_gemm_triton(a, bt, sa, sb, group_size=32):
    require_fp32(sa, sb)
    if (a.ndim != 2 or bt.ndim != 2 or a.shape[1] != bt.shape[1]
        or group_size not in (32, 64, 128)):
        raise ValueError("require A (M,K), BT (N,K), and supported G")
    m, k = a.shape
    n = bt.shape[0]
    if min(m, n, k) < 1 or max(a.numel(), bt.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions and 32-bit-safe offsets")
    if any(t.dtype != torch.int8 or t.device != sa.device or not t.is_contiguous() for t in (a, bt)):
        raise ValueError("A and BT must be contiguous INT8 on the scale device")
    groups = triton.cdiv(k, group_size)
    if sa.shape != (m, groups) or sb.shape != (n, groups):
        raise ValueError("scale matrices must have one value per row and K group")
    out = torch.empty((m, n), device=a.device)
    group_scaled_gemm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        a, bt, sa, sb, out, m, n, k, group_size, 32, 32, num_warps=4)
    return out


qx, sx = quantize_groups_triton(x, g)
qw, sw = quantize_groups_triton(w, g)
y = grouped_gemm_triton(qx, qw, sx, sw, g)


# 6. Compare zero groups, rounding ties, and M/N/K tails
for m, n, k, group_size in ((1, 1, 1, 32), (17, 19, 70, 32), (33, 35, 129, 64), (5, 7, 131, 128)):
    a = torch.randn(m, k, device=DEVICE)
    b = torch.randn(n, k, device=DEVICE)
    a[0] = 0
    aq, sa = quantize_groups_triton(a, group_size)
    bq, sb = quantize_groups_triton(b, group_size)
    rq, rs = quantize_groups_torch(a, group_size)
    torch.testing.assert_close(aq, rq, rtol=0, atol=0)
    torch.testing.assert_close(sa, rs, rtol=1e-6, atol=1e-7)
    da = dequantize_groups_triton(aq, sa, group_size)
    db = dequantize_groups_torch(bq, sb, group_size)
    torch.testing.assert_close(da, dequantize_groups_torch(aq, sa, group_size))
    torch.testing.assert_close(grouped_gemm_triton(aq, bq, sa, sb, group_size),
                               da @ db.T, rtol=2e-4, atol=2e-4)
ties = torch.tensor([[127., 0.5, -0.5, 1.5, -1.5]], device=DEVICE)
encoded, _ = quantize_groups_triton(ties)
assert encoded.cpu().tolist() == [[127, 1, -1, 2, -2]]
print("Group scales, zero rows, rounding ties, and tile tails passed.")


# 7. Benchmark prequantized GEMM
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


sample_x = torch.randn(128, 256, device=DEVICE)
sample_w = torch.randn(192, 256, device=DEVICE)
aq, sa = quantize_groups_triton(sample_x)
bq, sb = quantize_groups_triton(sample_w)
da, db = dequantize_groups_torch(aq, sa, 32), dequantize_groups_torch(bq, sb, 32)
torch.testing.assert_close(grouped_gemm_triton(aq, bq, sa, sb), da @ db.T, rtol=2e-4, atol=3e-4)
p = bench_ms(lambda: da @ db.T)
t = bench_ms(lambda: grouped_gemm_triton(aq, bq, sa, sb))
print(f"Reconstructed FP32 GEMM {p*1000:.3f} us; grouped INT8 GEMM {t*1000:.3f} us")


# 8. Measure representation error separately
approximate = grouped_gemm_triton(aq, bq, sa, sb)
exact = sample_x @ sample_w.T
relative_error = (approximate - exact).norm() / exact.norm().clamp_min(1e-12)
print("Relative error against original FP32 inputs:", relative_error.item())
```
