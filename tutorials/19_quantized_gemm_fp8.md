# 19. Quantized GEMM / FP8: low-precision products with explicit scales

Previous: [Quantization / dequantization](18_quantization_dequantization.md). Next: [RoPE](20_rope.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. The main path uses INT8 inputs with INT32 accumulation and an FP32 output. An optional native E4M3 FP8 path uses FP32 accumulation and requires a compatible PyTorch/Triton build and an NVIDIA GPU with compute capability >= 9.0; this guide deliberately restricts FP8 to Hopper or newer. All original inputs are finite, moderate, contiguous FP32 matrices. This is inference-only code without a registered backward operation.

## Concepts: move scaling out of the reduction

Let A have shape `(M,K)` and B have shape `(K,N)`. Quantize each row of A and each column of B, keeping scales constant over K:

$$
A_{m,k}\approx s^A_m A^q_{m,k},\qquad
B_{k,n}\approx s^B_n B^q_{k,n},\qquad
C_{m,n}\approx s^A_m s^B_n\sum_k A^q_{m,k}B^q_{k,n}.
$$

### 1. The dot product and the epilogue use different precision

INT8 multiplies signed integers and accumulates into INT32. Convert the accumulator to FP32, then multiply the two scales. This example restricts K to 65536, so even the worst-case sum of K products of values in `[-127,127]` fits INT32. Conversion to FP32 can round a large integer accumulator.

FP8 stores floating-point values with fewer exponent/mantissa bits. E4M3 has a largest finite magnitude of 448; its scale is based on that bound. The [Transformer Engine FP8 primer](https://nvidia.github.io/TransformerEngine/examples/fp8_primer.html) explains the formats and scaling tradeoffs. FP8 is a numerical format, and native FP8 matrix instructions also depend on hardware and compiler support.

### 2. Store the second operand as rows of B transpose

`bt_q` has shape `(N,K)` and contiguous storage. A program loads `a_q[m,k]` and `bt_q[n,k]`, constructing a `(BLOCK_K,BLOCK_N)` logical B tile through pointer arithmetic. The quantized tensors have one-byte elements, and the scales have FP32 elements.

```text
FP32 A -> row encoding -> A_q (M,K), s_A (M)
FP32 B -> transpose + row encoding -> BT_q (N,K), s_B (N)
                   |
           INT8 or FP8 tiled dot
                   |
          scale output once -> FP32 C
```

### 3. Keep quantization error separate from kernel error

The primary correctness reference multiplies the **same encoded values** in FP32 and applies the same scales. Comparing against the original `A @ B` measures approximation error as well. A faster kernel does not restore information removed by quantization. Per-K-group scales require applying different scale products inside the K loop, so the single epilogue here cannot implement block-scaled GEMM unchanged.

The [Triton dot API](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) describes input and accumulator precision. For native FP8 we set `max_num_imprecise_acc=0` to request FP32 accumulation without an interval of lower-precision accumulation. No `torch._scaled_mm` private API is required.

## 1. Imports and a small matrix product

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

a = torch.tensor([[1., 2., 3.], [-1., 0., 1.]], device=DEVICE)
b = torch.tensor([[2., 1.], [0., -1.], [1., 2.]], device=DEVICE)
FP8 = getattr(torch, "float8_e4m3fn", None)
HAS_FP8 = FP8 is not None and torch.cuda.get_device_capability(DEVICE)[0] >= 9
```

INT8 examples run independently of the FP8 feature check.

## 2. PyTorch eager: establish the original FP32 result

```python
c_eager = a @ b
print("A @ B:\n", c_eager.cpu())
```

This is the unquantized result. It is useful for measuring approximation, not for isolating errors in a quantized kernel.

## 3. PyTorch functions: encode rows and multiply encoded inputs

```python
def encode_rows(x, fp8=False):
    bound = 448.0 if fp8 else 127.0
    scales = (x.abs().amax(dim=1) / bound).clamp_min(1e-8)
    z = (x / scales[:, None]).clamp(-bound, bound)
    if fp8:
        if not HAS_FP8:
            raise RuntimeError("The native FP8 example requires E4M3 support and compute capability >= 9.0")
        encoded = z.to(FP8)
    else:
        encoded = (z.sign() * torch.floor(z.abs() + 0.5)).to(torch.int8)
    return encoded.contiguous(), scales.contiguous()


def prepare_inputs(a, b, fp8=False):
    aq, sa = encode_rows(a, fp8)
    btq, sb = encode_rows(b.T.contiguous(), fp8)
    return aq, btq, sa, sb


def quantized_matmul_torch(aq, btq, sa, sb):
    return (aq.float() @ btq.float().T) * sa[:, None] * sb[None, :]


aq, btq, sa, sb = prepare_inputs(a, b)
reference = quantized_matmul_torch(aq, btq, sa, sb)
print("Quantized reference:\n", reference.cpu())
```

Encoding is outside the GEMM timing unless the benchmark explicitly says otherwise. Reuse encoded weights across inference calls when possible.

## 4. Triton: INT32 or FP32 accumulation and a scale epilogue

```python
@triton.jit
def quantized_gemm_kernel(A, BT, SA, SB, C, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, IS_FP8: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    if IS_FP8:
        acc = tl.zeros((BM, BN), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.int32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + kk
        left = tl.load(A + rows[:, None] * K + k[None, :],
                       (rows[:, None] < M) & (k[None, :] < K), other=0)
        right = tl.load(BT + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        if IS_FP8:
            acc = tl.dot(left, right, acc, max_num_imprecise_acc=0)
        else:
            acc = tl.dot(left, right, acc)
    scale_a = tl.load(SA + rows, rows < M, other=0)
    scale_b = tl.load(SB + cols, cols < N, other=0)
    result = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    tl.store(C + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))
```

One program owns each output tile. M, N, and K have independent masks; no dequantized matrix or output atomics are needed.

## 5. Python wrapper: validate dtype, scales, and overflow bounds

```python
def quantized_matmul_triton(aq, btq, sa, sb):
    if aq.ndim != 2 or btq.ndim != 2 or aq.shape[1] != btq.shape[1]:
        raise ValueError("require A_q (M,K) and BT_q (N,K)")
    m, k = aq.shape
    n = btq.shape[0]
    if min(m, n, k) < 1 or k > 65536 or max(aq.numel(), btq.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions, K <= 65536, and 32-bit-safe offsets")
    is_fp8 = FP8 is not None and aq.dtype == FP8
    if aq.dtype != btq.dtype or (aq.dtype != torch.int8 and not is_fp8):
        raise ValueError("both matrices must use INT8 or E4M3 FP8")
    if is_fp8 and torch.cuda.get_device_capability(aq.device)[0] < 9:
        raise RuntimeError("This FP8 path requires compute capability >= 9.0")
    if sa.shape != (m,) or sb.shape != (n,) or sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise ValueError("require FP32 scale vectors of lengths M and N")
    tensors = (aq, btq, sa, sb)
    if not aq.is_cuda or any(t.device != aq.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    out = torch.empty((m, n), device=aq.device, dtype=torch.float32)
    quantized_gemm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        aq, btq, sa, sb, out, m, n, k, is_fp8, 32, 32, 64, num_warps=4)
    return out
```

The wrapper assumes finite positive scales. Arbitrary INT8 input values of -128 still fit the stated K bound; encoding in this notebook uses the symmetric range [-127,127].

## 6. Compare kernel error and original-input error

```python
for fp8 in (False, True):
    if fp8 and not HAS_FP8:
        print("Skipping native FP8: this GPU/build does not meet the stated requirements.")
        continue
    for m, n, k in ((2, 2, 3), (1, 1, 1), (17, 19, 23), (33, 65, 129)):
        left = torch.randn(m, k, device=DEVICE)
        right = torch.randn(k, n, device=DEVICE) / k**0.5
        encoded = prepare_inputs(left, right, fp8)
        expected = quantized_matmul_torch(*encoded)
        actual = quantized_matmul_triton(*encoded)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
        original = left @ right
        rms = (actual - original).square().mean().sqrt().item()
        print(f"{'FP8' if fp8 else 'INT8'} {m,n,k}: kernel max error "
              f"{(actual-expected).abs().max().item():.3e}, original RMS error {rms:.3e}")
```

Tolerances cover these moderate test inputs, not every quantized workload. Cancellation, outliers, and scale granularity can change application accuracy.

## 7. Benchmark a prepared quantized GEMM

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

left = torch.randn(256, 512, device=DEVICE)
right = torch.randn(512, 256, device=DEVICE) / 512**0.5
for fp8 in (False, True):
    if fp8 and not HAS_FP8:
        continue
    encoded = prepare_inputs(left, right, fp8)
    expected = quantized_matmul_torch(*encoded)
    actual = quantized_matmul_triton(*encoded)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    p = bench_ms(lambda: quantized_matmul_torch(*encoded))
    t = bench_ms(lambda: quantized_matmul_triton(*encoded))
    print(f"{'FP8' if fp8 else 'INT8'}: dequantized PyTorch {p*1000:.3f} us, "
          f"Triton {t*1000:.3f} us; GEMM work/time {2*256*256*512/(t*1e-3)/1e12:.3f} TFLOP/s")
```

The PyTorch reference converts encoded inputs to FP32 each call. It is an explicit correctness baseline, not an optimized native INT8/FP8 library comparison. Both calls allocate outputs; quantization, compilation, checks, and printing are outside timing. Throughput uses the conventional 2MNK work estimate, not a hardware counter.

## 8. Measure encoding cost and the zero-input boundary

```python
zeros = torch.zeros_like(left)
encoded = prepare_inputs(zeros, right)
torch.testing.assert_close(quantized_matmul_triton(*encoded),
                           torch.zeros(256, 256, device=DEVICE), rtol=0, atol=0)
end_to_end = bench_ms(lambda: quantized_matmul_triton(*prepare_inputs(left, right)))
prepared = prepare_inputs(left, right)
gemm_only = bench_ms(lambda: quantized_matmul_triton(*prepared))
print(f"INT8 encode both inputs + GEMM: {end_to_end*1000:.3f} us")
print(f"INT8 prepared GEMM:            {gemm_only*1000:.3f} us")
```

This end-to-end example deliberately re-encodes both matrices. Real weight reuse and fused activation quantization change the cost model. Fixed tiles emphasize the algorithm; tune only after repeating numerical checks.

## Try next

Continue with [RoPE](20_rope.md), then [cache updates](21_kv_cache_updates.md) and [decode attention](22_decode_attention.md). Compare native low-precision libraries only with matching formats, scales, output dtype, and timing scope.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a small matrix product
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

a = torch.tensor([[1., 2., 3.], [-1., 0., 1.]], device=DEVICE)
b = torch.tensor([[2., 1.], [0., -1.], [1., 2.]], device=DEVICE)
FP8 = getattr(torch, "float8_e4m3fn", None)
HAS_FP8 = FP8 is not None and torch.cuda.get_device_capability(DEVICE)[0] >= 9


# 2. PyTorch eager: establish the original FP32 result
c_eager = a @ b
print("A @ B:\n", c_eager.cpu())


# 3. PyTorch functions: encode rows and multiply encoded inputs
def encode_rows(x, fp8=False):
    bound = 448.0 if fp8 else 127.0
    scales = (x.abs().amax(dim=1) / bound).clamp_min(1e-8)
    z = (x / scales[:, None]).clamp(-bound, bound)
    if fp8:
        if not HAS_FP8:
            raise RuntimeError("The native FP8 example requires E4M3 support and compute capability >= 9.0")
        encoded = z.to(FP8)
    else:
        encoded = (z.sign() * torch.floor(z.abs() + 0.5)).to(torch.int8)
    return encoded.contiguous(), scales.contiguous()


def prepare_inputs(a, b, fp8=False):
    aq, sa = encode_rows(a, fp8)
    btq, sb = encode_rows(b.T.contiguous(), fp8)
    return aq, btq, sa, sb


def quantized_matmul_torch(aq, btq, sa, sb):
    return (aq.float() @ btq.float().T) * sa[:, None] * sb[None, :]


aq, btq, sa, sb = prepare_inputs(a, b)
reference = quantized_matmul_torch(aq, btq, sa, sb)
print("Quantized reference:\n", reference.cpu())


# 4. Triton: INT32 or FP32 accumulation and a scale epilogue
@triton.jit
def quantized_gemm_kernel(A, BT, SA, SB, C, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, IS_FP8: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    if IS_FP8:
        acc = tl.zeros((BM, BN), tl.float32)
    else:
        acc = tl.zeros((BM, BN), tl.int32)
    for start in range(0, tl.cdiv(K, BK)):
        k = start * BK + kk
        left = tl.load(A + rows[:, None] * K + k[None, :],
                       (rows[:, None] < M) & (k[None, :] < K), other=0)
        right = tl.load(BT + cols[None, :] * K + k[:, None],
                        (cols[None, :] < N) & (k[:, None] < K), other=0)
        if IS_FP8:
            acc = tl.dot(left, right, acc, max_num_imprecise_acc=0)
        else:
            acc = tl.dot(left, right, acc)
    scale_a = tl.load(SA + rows, rows < M, other=0)
    scale_b = tl.load(SB + cols, cols < N, other=0)
    result = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    tl.store(C + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))


# 5. Python wrapper: validate dtype, scales, and overflow bounds
def quantized_matmul_triton(aq, btq, sa, sb):
    if aq.ndim != 2 or btq.ndim != 2 or aq.shape[1] != btq.shape[1]:
        raise ValueError("require A_q (M,K) and BT_q (N,K)")
    m, k = aq.shape
    n = btq.shape[0]
    if min(m, n, k) < 1 or k > 65536 or max(aq.numel(), btq.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions, K <= 65536, and 32-bit-safe offsets")
    is_fp8 = FP8 is not None and aq.dtype == FP8
    if aq.dtype != btq.dtype or (aq.dtype != torch.int8 and not is_fp8):
        raise ValueError("both matrices must use INT8 or E4M3 FP8")
    if is_fp8 and torch.cuda.get_device_capability(aq.device)[0] < 9:
        raise RuntimeError("This FP8 path requires compute capability >= 9.0")
    if sa.shape != (m,) or sb.shape != (n,) or sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise ValueError("require FP32 scale vectors of lengths M and N")
    tensors = (aq, btq, sa, sb)
    if not aq.is_cuda or any(t.device != aq.device or not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous on the same CUDA device")
    out = torch.empty((m, n), device=aq.device, dtype=torch.float32)
    quantized_gemm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        aq, btq, sa, sb, out, m, n, k, is_fp8, 32, 32, 64, num_warps=4)
    return out


# 6. Compare kernel error and original-input error
for fp8 in (False, True):
    if fp8 and not HAS_FP8:
        print("Skipping native FP8: this GPU/build does not meet the stated requirements.")
        continue
    for m, n, k in ((2, 2, 3), (1, 1, 1), (17, 19, 23), (33, 65, 129)):
        left = torch.randn(m, k, device=DEVICE)
        right = torch.randn(k, n, device=DEVICE) / k**0.5
        encoded = prepare_inputs(left, right, fp8)
        expected = quantized_matmul_torch(*encoded)
        actual = quantized_matmul_triton(*encoded)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
        original = left @ right
        rms = (actual - original).square().mean().sqrt().item()
        print(f"{'FP8' if fp8 else 'INT8'} {m,n,k}: kernel max error "
              f"{(actual-expected).abs().max().item():.3e}, original RMS error {rms:.3e}")


# 7. Benchmark a prepared quantized GEMM
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

left = torch.randn(256, 512, device=DEVICE)
right = torch.randn(512, 256, device=DEVICE) / 512**0.5
for fp8 in (False, True):
    if fp8 and not HAS_FP8:
        continue
    encoded = prepare_inputs(left, right, fp8)
    expected = quantized_matmul_torch(*encoded)
    actual = quantized_matmul_triton(*encoded)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    p = bench_ms(lambda: quantized_matmul_torch(*encoded))
    t = bench_ms(lambda: quantized_matmul_triton(*encoded))
    print(f"{'FP8' if fp8 else 'INT8'}: dequantized PyTorch {p*1000:.3f} us, "
          f"Triton {t*1000:.3f} us; GEMM work/time {2*256*256*512/(t*1e-3)/1e12:.3f} TFLOP/s")


# 8. Measure encoding cost and the zero-input boundary
zeros = torch.zeros_like(left)
encoded = prepare_inputs(zeros, right)
torch.testing.assert_close(quantized_matmul_triton(*encoded),
                           torch.zeros(256, 256, device=DEVICE), rtol=0, atol=0)
end_to_end = bench_ms(lambda: quantized_matmul_triton(*prepare_inputs(left, right)))
prepared = prepare_inputs(left, right)
gemm_only = bench_ms(lambda: quantized_matmul_triton(*prepared))
print(f"INT8 encode both inputs + GEMM: {end_to_end*1000:.3f} us")
print(f"INT8 prepared GEMM:            {gemm_only*1000:.3f} us")
```
