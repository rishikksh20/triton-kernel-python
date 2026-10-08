# 18. Quantization / dequantization: row scales and packed INT4

Previous: [FlashAttention backward](17_flashattention_backward.md). Next: [Quantized GEMM / FP8](19_quantized_gemm_fp8.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 matrices `(R,C)` on one NVIDIA CUDA GPU, with finite moderate values, R >= 1, and C=1–4096. The integer formats are symmetric signed INT8 and signed INT4. Quantization is lossy and these forward kernels have no registered PyTorch backward operation.

## Concepts: separate the stored integer from its scale

For each row r, choose a dequantization scale and encode each feature:

$$
s_r=\max\left(\frac{\max_c|x_{r,c}|}{Q},10^{-8}\right),\qquad
q_{r,c}=\operatorname{clip}\left(\operatorname{round}_{\rm away}(x_{r,c}/s_r),-Q,Q\right),\qquad
\widehat x_{r,c}=s_rq_{r,c}.
$$

`Q=127` for INT8 and `Q=7` for INT4. We explicitly round ties away from zero in both implementations. The scale floor makes zero rows safe and prevents division by an extremely small scale. It also means values much smaller than the floor can quantize to zero. This scale multiplies stored values during dequantization; it is the inverse of a quantization multiplier.

### 1. Reduce once, then broadcast within the row

One program loads a row, reduces its absolute maximum, computes its scale, and encodes the row. Padding for the maximum uses zero, and stores mask the padded columns. Dequantization broadcasts that same row scale.

### 2. Two INT4 values share one byte

INT4 uses two's-complement nibbles. The first value occupies bits 0–3 and the second occupies bits 4–7. `-1` becomes nibble `15`; `-7` becomes `9`. On unpacking, subtract 16 from a nibble >= 8. Our symmetric scheme leaves `-8` unused.

```text
FP32 row -> max(abs(x)) -> scale -> signed integers
INT8:   one integer per byte
INT4:   q[2*j] in low nibble | q[2*j+1] in high nibble
decode: byte -> signed nibble -> multiply by row scale
```

An odd last feature uses a zero high nibble. The original column count must travel with packed data; the byte tensor alone cannot tell whether its final high nibble is padding. The INT4 implementation uses an intermediate INT8 matrix to make encoding and packing easy to inspect. Fuse those stages after validating their separate behavior.

### 3. Check representation and approximation separately

Integer bytes and scales should match the reference encoding. Reconstructed FP32 values should match the reference decoding. Reconstruction need not equal the original tensor: ignoring clipping and floating-point effects, its rounding error is at most half a row scale. Outliers enlarge that scale and reduce resolution for every other value in the row.

## 1. Imports and rows with different ranges

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

x = torch.tensor([[0., 0., 0., 0., 0.], [-7., -1., 0., 1., 7.],
                  [-0.3, 0.1, 0.2, 0.4, 0.7]], device=DEVICE)
```

The zero row and odd column count exercise two representation boundaries.

## 2. PyTorch eager: inspect an INT8 encoding

```python
scale = (x.abs().amax(dim=1) / 127).clamp_min(1e-8)
z = x / scale[:, None]
q = (z.sign() * torch.floor(z.abs() + 0.5)).clamp(-127, 127).to(torch.int8)
print("Scales:", scale.cpu())
print("Integers:", q.cpu())
print("Reconstructed:\n", (q.float() * scale[:, None]).cpu())
```

The integers describe levels; the scale describes the spacing between levels.

## 3. PyTorch functions: encode, pack, and decode

```python
def quantize_torch(x, bits=8):
    limit = 127 if bits == 8 else 7
    scale = (x.abs().amax(dim=1) / limit).clamp_min(1e-8)
    z = x / scale[:, None]
    q = (z.sign() * torch.floor(z.abs() + 0.5)).clamp(-limit, limit).to(torch.int8)
    if bits == 8:
        return q, scale
    low = q[:, 0::2].to(torch.int16) & 15
    high = torch.zeros_like(low)
    high[:, :q[:, 1::2].shape[1]] = q[:, 1::2].to(torch.int16) & 15
    return (low | (high << 4)).to(torch.uint8), scale


def dequantize_torch(data, scale, columns, bits=8):
    if bits == 8:
        return data.float() * scale[:, None]
    nibble = torch.empty((data.shape[0], 2 * data.shape[1]), device=data.device, dtype=torch.int16)
    nibble[:, 0::2] = data.to(torch.int16) & 15
    nibble[:, 1::2] = (data.to(torch.int16) >> 4) & 15
    signed = torch.where(nibble >= 8, nibble - 16, nibble)
    return signed[:, :columns].float() * scale[:, None]
```

The packed result has shape `(R,ceil(C/2))` and dtype uint8. A uint8 container stores bit patterns, not unsigned numerical INT4 values.

## 4. Triton: row reduction, byte packing, and decoding

```python
@triton.jit
def quantize_kernel(X, Q, S, C: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=0)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), LIMIT), 1e-8)
    z = tl.div_rn(x, scale)
    rounded = tl.where(z < 0, -tl.floor(-z + 0.5), tl.floor(z + 0.5))
    q = tl.minimum(tl.maximum(rounded, -LIMIT), LIMIT).to(tl.int8)
    tl.store(Q + row * C + col, q, col < C)
    tl.store(S + row, scale)


@triton.jit
def pack_kernel(Q, P, C: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    low = tl.load(Q + row * C + 2 * j, 2 * j < C, other=0).to(tl.int32) & 15
    high = tl.load(Q + row * C + 2 * j + 1, 2 * j + 1 < C, other=0).to(tl.int32) & 15
    tl.store(P + row * PACKED + j, low | (high << 4), j < PACKED)


@triton.jit
def dequantize_kernel(P, S, Y, C: tl.constexpr, STRIDE: tl.constexpr,
                      BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if BITS == 8:
        q = tl.load(P + row * STRIDE + col, col < C, other=0).to(tl.float32)
    else:
        byte = tl.load(P + row * STRIDE + col // 2, col < C, other=0).to(tl.int32)
        nibble = (byte >> (4 * (col % 2))) & 15
        q = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    scale = tl.load(S + row)
    tl.store(Y + row * C + col, q * scale, col < C)
```

Quantization uses [precise FP32 division](https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html) so approximate reciprocals cannot change encoding near a half-step boundary. Quantization and dequantization each own a row. Packing reads two source integers per output byte; no two programs write the same byte.

## 5. Python wrappers: preserve format metadata

```python
def quantize_triton(x, bits=8):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA FP32 matrix")
    rows, columns = x.shape
    if rows < 1 or not 1 <= columns <= 4096 or x.numel() >= 2**31:
        raise ValueError("require R >= 1, 1 <= C <= 4096, and fewer than 2**31 elements")
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    quantize_kernel[(rows,)](x, q, scale, columns, 127 if bits == 8 else 7,
                            triton.next_power_of_2(columns), num_warps=4)
    if bits == 8:
        return q, scale
    packed_columns = triton.cdiv(columns, 2)
    packed = torch.empty((rows, packed_columns), device=x.device, dtype=torch.uint8)
    pack_kernel[(rows,)](q, packed, columns, packed_columns,
                        triton.next_power_of_2(packed_columns), num_warps=4)
    return packed, scale


def dequantize_triton(data, scale, columns, bits=8):
    if bits not in (4, 8) or not 1 <= columns <= 4096:
        raise ValueError("invalid bits or original column count")
    dtype = torch.int8 if bits == 8 else torch.uint8
    width = columns if bits == 8 else triton.cdiv(columns, 2)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != width or data.dtype != dtype:
        raise ValueError("data shape/dtype does not match format metadata")
    if scale.shape != (data.shape[0],) or scale.dtype != torch.float32:
        raise ValueError("scale must be an FP32 vector with one value per row")
    if not data.is_cuda or scale.device != data.device or not data.is_contiguous() or not scale.is_contiguous():
        raise ValueError("data and scales must be contiguous on the same CUDA device")
    if data.shape[0] * columns >= 2**31:
        raise ValueError("decoded tensor must have fewer than 2**31 elements")
    out = torch.empty((data.shape[0], columns), device=data.device, dtype=torch.float32)
    dequantize_kernel[(data.shape[0],)](data, scale, out, columns, width, bits,
                                      triton.next_power_of_2(columns), num_warps=4)
    return out
```

Scale values are assumed finite and positive. The wrapper checks metadata before decoding so an odd-width packed tensor cannot silently gain a feature.

## 6. Compare bytes, reconstruction, and tail columns

```python
for bits in (8, 4):
    for rows, columns in ((3, 5), (1, 1), (7, 33), (4, 127), (2, 4096)):
        sample = torch.randn(rows, columns, device=DEVICE)
        sample[0].zero_()
        expected, ref_scale = quantize_torch(sample, bits)
        actual, scale = quantize_triton(sample, bits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-8)
        decoded = dequantize_triton(actual, scale, columns, bits)
        reference = dequantize_torch(expected, ref_scale, columns, bits)
        torch.testing.assert_close(decoded, reference, rtol=1e-6, atol=1e-6)
        assert (decoded - sample).abs().le(scale[:, None] * 0.501 + 1e-6).all().item()
print("INT8/INT4 encoding, zero-row, packing, and tail checks passed.")
```

Random inputs rarely land exactly on a rounding boundary. Section 8 separately checks exact half-step values.

## 7. Benchmark encoding and decoding separately

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(256, 1023, device=DEVICE)
for bits in (8, 4):
    data, scales = quantize_triton(sample, bits)
    for name, torch_fn, triton_fn in (
        ("encode", lambda: quantize_torch(sample, bits), lambda: quantize_triton(sample, bits)),
        ("decode", lambda: dequantize_torch(data, scales, 1023, bits),
                   lambda: dequantize_triton(data, scales, 1023, bits)),
    ):
        p, t = bench_ms(torch_fn), bench_ms(triton_fn)
        print(f"INT{bits} {name}: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, {p/t:.2f}x")
```

These allocating calls include the extra INT4 packing launch. GPU event medians are printed in microseconds; input creation, compilation, correctness checks, and printing are outside timing. No speedup is assumed.

## 8. Inspect half-step rounding and compression

```python
for bits, limit in ((8, 127), (4, 7)):
    sample = torch.tensor([[float(limit), 0.5, -0.5, 1.5, -1.5]], device=DEVICE)
    data, scales = quantize_triton(sample, bits)
    decoded = dequantize_triton(data, scales, 5, bits)
    torch.testing.assert_close(decoded, torch.tensor([[float(limit), 1., -1., 2., -2.]], device=DEVICE))
    storage = data.numel() * data.element_size() + scales.numel() * scales.element_size()
    original = sample.numel() * sample.element_size()
    print(f"INT{bits}: {storage} bytes including scales, versus {original} original bytes")
```

Small rows can spend much of their storage on scales. The INT4 measurement counts the final representation, excluding the temporary INT8 workspace used while encoding.

## Try next

Continue with [quantized GEMM / FP8](19_quantized_gemm_fp8.md). Try per-group scales and fused packing, while keeping bit order, rounding, and scale conventions explicit.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and rows with different ranges
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

x = torch.tensor([[0., 0., 0., 0., 0.], [-7., -1., 0., 1., 7.],
                  [-0.3, 0.1, 0.2, 0.4, 0.7]], device=DEVICE)


# 2. PyTorch eager: inspect an INT8 encoding
scale = (x.abs().amax(dim=1) / 127).clamp_min(1e-8)
z = x / scale[:, None]
q = (z.sign() * torch.floor(z.abs() + 0.5)).clamp(-127, 127).to(torch.int8)
print("Scales:", scale.cpu())
print("Integers:", q.cpu())
print("Reconstructed:\n", (q.float() * scale[:, None]).cpu())


# 3. PyTorch functions: encode, pack, and decode
def quantize_torch(x, bits=8):
    limit = 127 if bits == 8 else 7
    scale = (x.abs().amax(dim=1) / limit).clamp_min(1e-8)
    z = x / scale[:, None]
    q = (z.sign() * torch.floor(z.abs() + 0.5)).clamp(-limit, limit).to(torch.int8)
    if bits == 8:
        return q, scale
    low = q[:, 0::2].to(torch.int16) & 15
    high = torch.zeros_like(low)
    high[:, :q[:, 1::2].shape[1]] = q[:, 1::2].to(torch.int16) & 15
    return (low | (high << 4)).to(torch.uint8), scale


def dequantize_torch(data, scale, columns, bits=8):
    if bits == 8:
        return data.float() * scale[:, None]
    nibble = torch.empty((data.shape[0], 2 * data.shape[1]), device=data.device, dtype=torch.int16)
    nibble[:, 0::2] = data.to(torch.int16) & 15
    nibble[:, 1::2] = (data.to(torch.int16) >> 4) & 15
    signed = torch.where(nibble >= 8, nibble - 16, nibble)
    return signed[:, :columns].float() * scale[:, None]


# 4. Triton: row reduction, byte packing, and decoding
@triton.jit
def quantize_kernel(X, Q, S, C: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=0)
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), LIMIT), 1e-8)
    z = tl.div_rn(x, scale)
    rounded = tl.where(z < 0, -tl.floor(-z + 0.5), tl.floor(z + 0.5))
    q = tl.minimum(tl.maximum(rounded, -LIMIT), LIMIT).to(tl.int8)
    tl.store(Q + row * C + col, q, col < C)
    tl.store(S + row, scale)


@triton.jit
def pack_kernel(Q, P, C: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    low = tl.load(Q + row * C + 2 * j, 2 * j < C, other=0).to(tl.int32) & 15
    high = tl.load(Q + row * C + 2 * j + 1, 2 * j + 1 < C, other=0).to(tl.int32) & 15
    tl.store(P + row * PACKED + j, low | (high << 4), j < PACKED)


@triton.jit
def dequantize_kernel(P, S, Y, C: tl.constexpr, STRIDE: tl.constexpr,
                      BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    if BITS == 8:
        q = tl.load(P + row * STRIDE + col, col < C, other=0).to(tl.float32)
    else:
        byte = tl.load(P + row * STRIDE + col // 2, col < C, other=0).to(tl.int32)
        nibble = (byte >> (4 * (col % 2))) & 15
        q = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    scale = tl.load(S + row)
    tl.store(Y + row * C + col, q * scale, col < C)


# 5. Python wrappers: preserve format metadata
def quantize_triton(x, bits=8):
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if x.ndim != 2 or x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA FP32 matrix")
    rows, columns = x.shape
    if rows < 1 or not 1 <= columns <= 4096 or x.numel() >= 2**31:
        raise ValueError("require R >= 1, 1 <= C <= 4096, and fewer than 2**31 elements")
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(rows, device=x.device, dtype=torch.float32)
    quantize_kernel[(rows,)](x, q, scale, columns, 127 if bits == 8 else 7,
                            triton.next_power_of_2(columns), num_warps=4)
    if bits == 8:
        return q, scale
    packed_columns = triton.cdiv(columns, 2)
    packed = torch.empty((rows, packed_columns), device=x.device, dtype=torch.uint8)
    pack_kernel[(rows,)](q, packed, columns, packed_columns,
                        triton.next_power_of_2(packed_columns), num_warps=4)
    return packed, scale


def dequantize_triton(data, scale, columns, bits=8):
    if bits not in (4, 8) or not 1 <= columns <= 4096:
        raise ValueError("invalid bits or original column count")
    dtype = torch.int8 if bits == 8 else torch.uint8
    width = columns if bits == 8 else triton.cdiv(columns, 2)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] != width or data.dtype != dtype:
        raise ValueError("data shape/dtype does not match format metadata")
    if scale.shape != (data.shape[0],) or scale.dtype != torch.float32:
        raise ValueError("scale must be an FP32 vector with one value per row")
    if not data.is_cuda or scale.device != data.device or not data.is_contiguous() or not scale.is_contiguous():
        raise ValueError("data and scales must be contiguous on the same CUDA device")
    if data.shape[0] * columns >= 2**31:
        raise ValueError("decoded tensor must have fewer than 2**31 elements")
    out = torch.empty((data.shape[0], columns), device=data.device, dtype=torch.float32)
    dequantize_kernel[(data.shape[0],)](data, scale, out, columns, width, bits,
                                      triton.next_power_of_2(columns), num_warps=4)
    return out


# 6. Compare bytes, reconstruction, and tail columns
for bits in (8, 4):
    for rows, columns in ((3, 5), (1, 1), (7, 33), (4, 127), (2, 4096)):
        sample = torch.randn(rows, columns, device=DEVICE)
        sample[0].zero_()
        expected, ref_scale = quantize_torch(sample, bits)
        actual, scale = quantize_triton(sample, bits)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=1e-8)
        decoded = dequantize_triton(actual, scale, columns, bits)
        reference = dequantize_torch(expected, ref_scale, columns, bits)
        torch.testing.assert_close(decoded, reference, rtol=1e-6, atol=1e-6)
        assert (decoded - sample).abs().le(scale[:, None] * 0.501 + 1e-6).all().item()
print("INT8/INT4 encoding, zero-row, packing, and tail checks passed.")


# 7. Benchmark encoding and decoding separately
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(256, 1023, device=DEVICE)
for bits in (8, 4):
    data, scales = quantize_triton(sample, bits)
    for name, torch_fn, triton_fn in (
        ("encode", lambda: quantize_torch(sample, bits), lambda: quantize_triton(sample, bits)),
        ("decode", lambda: dequantize_torch(data, scales, 1023, bits),
                   lambda: dequantize_triton(data, scales, 1023, bits)),
    ):
        p, t = bench_ms(torch_fn), bench_ms(triton_fn)
        print(f"INT{bits} {name}: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, {p/t:.2f}x")


# 8. Inspect half-step rounding and compression
for bits, limit in ((8, 127), (4, 7)):
    sample = torch.tensor([[float(limit), 0.5, -0.5, 1.5, -1.5]], device=DEVICE)
    data, scales = quantize_triton(sample, bits)
    decoded = dequantize_triton(data, scales, 5, bits)
    torch.testing.assert_close(decoded, torch.tensor([[float(limit), 1., -1., 2., -2.]], device=DEVICE))
    storage = data.numel() * data.element_size() + scales.numel() * scales.element_size()
    original = sample.numel() * sample.element_size()
    print(f"INT{bits}: {storage} bytes including scales, versus {original} original bytes")
```
