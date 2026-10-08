# 8. LayerNorm: center and scale each feature row

Previous: [RMSNorm](07_rmsnorm.md). Next: [MatMul](09_matmul.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. The examples normalize the last dimension of a contiguous **FP32 matrix on one NVIDIA CUDA GPU**, with 1–4096 features, and use contiguous FP32 weight and bias vectors. Inputs and parameters are finite, and their reductions must remain representable in FP32. The kernels implement forward computation without a registered PyTorch backward operation.

## The operation and its equation

Layer normalization computes a mean and variance within each row, then applies a feature-wise scale $\gamma$ and shift $\beta$:

$$
\mu_r=\frac{1}{N}\sum_{j=0}^{N-1}x_{r,j},\qquad
v_r=\frac{1}{N}\sum_{j=0}^{N-1}(x_{r,j}-\mu_r)^2,
$$

$$
y_{r,c}=\gamma_c\frac{x_{r,c}-\mu_r}{\sqrt{v_r+\varepsilon}}+\beta_c.
$$

The variance divides by **$N$**, not $N-1$. This is the population, or biased, variance used by [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html). Normalization is over features within a row, rather than across the batch.

For `x = [1, 2, 3]`, the mean is `2` and the variance is `2/3`. Before the affine scale and bias, the output with `eps=1e-5` is approximately `[-1.224736, 0, 1.224736]`. With `gamma=[1,2,0.5]` and `beta=[0.1,-0.2,0.3]`, it becomes approximately `[-1.124736,-0.2,0.912368]`.

A constant row has zero centered values, so its affine output is exactly its bias vector. A one-feature row has the same behavior.

```mermaid
flowchart LR
    X[Load row] --> M[Mean]
    X --> C[Subtract mean]
    M --> C
    C --> V[Mean of centered squares]
    V --> R[Add epsilon and take rsqrt]
    C --> S[Multiply by row scale]
    R --> S
    S --> A[Apply feature weight and bias]
    A --> Y[Store row]
```

| Property | RMSNorm | LayerNorm |
| --- | --- | --- |
| Row statistic | Mean square of input | Mean and centered variance |
| Subtract row mean | No | Yes |
| Affine parameters here | Weight | Weight and bias |
| Constant nonzero input before affine | Generally remains nonzero | Becomes zero |

## 1. Imports, epsilon, and example parameters

Use a CUDA-enabled PyTorch build and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")

DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
EPS = 1e-5
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 2., 3.], [5., 5., 5.]], device=DEVICE, dtype=DTYPE)
weight = torch.tensor([1., 2., 0.5], device=DEVICE, dtype=DTYPE)
bias = torch.tensor([0.1, -0.2, 0.3], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

## 2. PyTorch eager: expand mean, variance, and affine steps

```python
mean = x.mean(dim=-1, keepdim=True)
centered = x - mean
variance = centered.square().mean(dim=-1, keepdim=True)
normalized = centered * torch.rsqrt(variance + EPS)
y_formula = normalized * weight + bias
y_eager = F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=EPS)
torch.testing.assert_close(y_formula, y_eager, rtol=1e-5, atol=1e-5)
print("Mean per row:", mean.cpu().flatten().tolist())
print("Variance per row:", variance.cpu().flatten().tolist())
print("LayerNorm output:\n", y_eager.cpu())
```

1. Find each row's mean and subtract it from the input.
2. Average the centered squares to obtain the population variance.
3. Add epsilon inside the reciprocal square root and normalize the centered input.
4. Multiply by feature weights and add feature biases.

If you replace the explicit variance with `torch.var`, use `correction=0` to preserve this equation. PyTorch's default sample-variance correction would change the operation.

## 3. PyTorch functions: built-in and expanded references

```python
def layernorm_torch(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def layernorm_expanded(x, weight, bias, eps=1e-5):
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(variance + eps)) * weight + bias


torch.testing.assert_close(layernorm_torch(x, weight, bias, EPS), y_eager, rtol=0, atol=0)
torch.testing.assert_close(layernorm_expanded(x, weight, bias, EPS), y_eager, rtol=1e-5, atol=1e-5)
```

These functions still execute eagerly. The expanded formula exposes the intermediate tensors; the built-in is the primary performance reference.

## 4. Triton: mean, centered variance, and affine transform

```python
@triton.jit
def layernorm_kernel(
    X, Weight, Bias, Y, EPS,
    N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / N_COLS

    # Padding must stay zero after centering, not become -mean.
    centered = tl.where(valid, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N_COLS
    normalized = centered * tl.rsqrt(variance + EPS)

    weights = tl.load(Weight + columns, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(Bias + columns, mask=valid, other=0.0).to(tl.float32)
    result = normalized * weights + biases
    tl.store(Y + row * N_COLS + columns, result, mask=valid)
```

1. Load one padded row and compute its mean using the real feature count.
2. Center valid values and explicitly keep padded values at zero.
3. Reduce the centered squares, again dividing by the real feature count.
4. Normalize, apply weights and biases, and store valid output columns.

Without `tl.where(valid, ..., 0.0)`, initially padded zeros would become `-mean` after centering and contribute an incorrect squared term. This is a different padding requirement from RMSNorm.

Computing centered squares avoids the subtraction of two large, nearly equal quantities in `mean(x*x) - mean(x)**2`. PyTorch may use a different statistics algorithm, so results still require floating-point tolerances. The [official Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) explains the row-wise approach and also covers backward kernels.

## 5. Python wrapper: validate parameters and row width

```python
def layernorm_triton(x, weight, bias, eps=1e-5):
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda and x.device == weight.device == bias.device):
        raise ValueError("Input, weight, and bias must be CUDA tensors on the same device.")
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("Expected a matrix and two parameter vectors.")
    rows, columns = x.shape
    if not 1 <= columns <= 4096 or weight.numel() != columns or bias.numel() != columns:
        raise ValueError("Use 1–4096 features and one weight/bias per feature.")
    if any(t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError("This tutorial supports FP32 inputs and parameters.")
    if not all(t.is_contiguous() for t in (x, weight, bias)):
        raise ValueError("Input and parameter vectors must be contiguous.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Epsilon must be positive and finite.")
    y = torch.empty_like(x)
    if rows == 0:
        return y
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size <= 1024 else 8
    with torch.cuda.device(x.device):
        layernorm_kernel[(rows,)](
            x, weight, bias, y, float(eps), N_COLS=columns,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
    return y
```

The wrapper returns an empty output for zero rows and rejects zero features. The 4096-feature limit bounds the teaching kernel's local work. The affine vectors are shared across rows, and epsilon is passed explicitly to both implementations.

## 6. Compare values, constant rows, and non-power-of-two widths

```python
y_triton = layernorm_triton(x, weight, bias, EPS)
torch.testing.assert_close(y_triton, y_eager, rtol=1e-5, atol=1e-5)
print("row, column | PyTorch    | Triton     | abs error")
for r, (references, results) in enumerate(zip(y_eager.cpu().tolist(), y_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:10.6f} | {got:10.6f} | {abs(ref-got):.3e}")

for rows, columns in ((0, 7), (1, 1), (3, 7), (5, 257), (17, 1003), (4, 4096)):
    values = torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    weights = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    biases = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    if rows > 0:
        values[0] = 5.0  # Constant row: normalized values are zero.
    actual = layernorm_triton(values, weights, biases, EPS)
    expected = layernorm_torch(values, weights, biases, EPS)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    if rows > 0:
        torch.testing.assert_close(actual[0], biases, rtol=0, atol=0)
print("All value, constant-row, affine, padding, and empty-batch checks passed.")
```

Widths such as `7`, `257`, and `1003` expose padding errors. A one-feature row must produce its bias, independent of its input or weight. For demanding numerical cases, such as large means with tiny variance, also examine FP64 diagnostics and choose a statistics algorithm and tolerances for the application.

## 7. Compare eager, function, expanded, and Triton times

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("shape       eager us  function us  expanded us  Triton us  speedup  max error")
for rows, columns in ((32, 32), (256, 257), (1024, 1024), (1024, 4096)):
    values = torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    weights = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    biases = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    expected = layernorm_torch(values, weights, biases, EPS)
    actual = layernorm_triton(values, weights, biases, EPS)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: F.layer_norm(values, (columns,), weight=weights, bias=biases, eps=EPS))
    function_ms = bench_ms(lambda: layernorm_torch(values, weights, biases, EPS))
    expanded_ms = bench_ms(lambda: layernorm_expanded(values, weights, biases, EPS))
    triton_ms = bench_ms(lambda: layernorm_triton(values, weights, biases, EPS))
    shape = f"{rows}x{columns}"
    print(
        f"{shape:11s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{expanded_ms*1000:12.3f} {triton_ms*1000:10.3f} "
        f"{function_ms/triton_ms:7.2f}x {error:10.3e}"
    )
```

The same input, weights, biases, dtype, and epsilon are used for every path. Every callable allocates an output; the expanded formula also allocates intermediate tensors. The speedup compares Triton with the built-in PyTorch function rather than only with the expanded formula.

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds, converted here to median microseconds. These GPU event timings can include host dispatch gaps. Input creation, compilation, correctness checks, CPU copies, and printing are outside timing. Full CPU wall-clock time is a separate measurement. Run the block for your own GPU timings; the simple kernel may be slower than the optimized built-in.

## 8. Check statistics before affine scale and bias

```python
values = torch.tensor([[1., 2., 3.], [5., 5., 5.]], device=DEVICE, dtype=DTYPE)
unit_weight = torch.ones(3, device=DEVICE, dtype=DTYPE)
zero_bias = torch.zeros(3, device=DEVICE, dtype=DTYPE)
normalized = layernorm_triton(values, unit_weight, zero_bias, EPS)
centered = values - values.mean(dim=-1, keepdim=True)
variance = centered.square().mean(dim=-1)
expected_variance = variance / (variance + EPS)
torch.testing.assert_close(
    normalized.mean(dim=-1), torch.zeros(2, device=DEVICE, dtype=DTYPE),
    rtol=0, atol=1e-6,
)
torch.testing.assert_close(
    normalized.square().mean(dim=-1), expected_variance, rtol=1e-5, atol=1e-6
)
print("Without affine parameters:\n", normalized.cpu())
print("Output mean per row:", normalized.mean(dim=-1).cpu().tolist())
print("Output variance per row:", normalized.var(dim=-1, correction=0).cpu().tolist())
```

Before affine parameters, the row mean is approximately zero and the variance is $v/(v+\varepsilon)$, not exactly one when epsilon matters. Feature-dependent weights and biases can change both statistics. The constant row has zero variance and remains zero in this unweighted example.

## Try next

Continue with [MatMul](09_matmul.md), where each output depends on a dot product and programs own two-dimensional tiles.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports, epsilon, and example parameters
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")

DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
EPS = 1e-5
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 2., 3.], [5., 5., 5.]], device=DEVICE, dtype=DTYPE)
weight = torch.tensor([1., 2., 0.5], device=DEVICE, dtype=DTYPE)
bias = torch.tensor([0.1, -0.2, 0.3], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: expand mean, variance, and affine steps
mean = x.mean(dim=-1, keepdim=True)
centered = x - mean
variance = centered.square().mean(dim=-1, keepdim=True)
normalized = centered * torch.rsqrt(variance + EPS)
y_formula = normalized * weight + bias
y_eager = F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=EPS)
torch.testing.assert_close(y_formula, y_eager, rtol=1e-5, atol=1e-5)
print("Mean per row:", mean.cpu().flatten().tolist())
print("Variance per row:", variance.cpu().flatten().tolist())
print("LayerNorm output:\n", y_eager.cpu())


# 3. PyTorch functions: built-in and expanded references
def layernorm_torch(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), weight=weight, bias=bias, eps=eps)


def layernorm_expanded(x, weight, bias, eps=1e-5):
    centered = x - x.mean(dim=-1, keepdim=True)
    variance = centered.square().mean(dim=-1, keepdim=True)
    return (centered * torch.rsqrt(variance + eps)) * weight + bias


torch.testing.assert_close(layernorm_torch(x, weight, bias, EPS), y_eager, rtol=0, atol=0)
torch.testing.assert_close(layernorm_expanded(x, weight, bias, EPS), y_eager, rtol=1e-5, atol=1e-5)


# 4. Triton: mean, centered variance, and affine transform
@triton.jit
def layernorm_kernel(
    X, Weight, Bias, Y, EPS,
    N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=0.0).to(tl.float32)
    mean = tl.sum(values, axis=0) / N_COLS

    # Padding must stay zero after centering, not become -mean.
    centered = tl.where(valid, values - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / N_COLS
    normalized = centered * tl.rsqrt(variance + EPS)

    weights = tl.load(Weight + columns, mask=valid, other=0.0).to(tl.float32)
    biases = tl.load(Bias + columns, mask=valid, other=0.0).to(tl.float32)
    result = normalized * weights + biases
    tl.store(Y + row * N_COLS + columns, result, mask=valid)


# 5. Python wrapper: validate parameters and row width
def layernorm_triton(x, weight, bias, eps=1e-5):
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda and x.device == weight.device == bias.device):
        raise ValueError("Input, weight, and bias must be CUDA tensors on the same device.")
    if x.ndim != 2 or weight.ndim != 1 or bias.ndim != 1:
        raise ValueError("Expected a matrix and two parameter vectors.")
    rows, columns = x.shape
    if not 1 <= columns <= 4096 or weight.numel() != columns or bias.numel() != columns:
        raise ValueError("Use 1–4096 features and one weight/bias per feature.")
    if any(t.dtype != torch.float32 for t in (x, weight, bias)):
        raise ValueError("This tutorial supports FP32 inputs and parameters.")
    if not all(t.is_contiguous() for t in (x, weight, bias)):
        raise ValueError("Input and parameter vectors must be contiguous.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Epsilon must be positive and finite.")
    y = torch.empty_like(x)
    if rows == 0:
        return y
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size <= 1024 else 8
    with torch.cuda.device(x.device):
        layernorm_kernel[(rows,)](
            x, weight, bias, y, float(eps), N_COLS=columns,
            BLOCK_SIZE=block_size, num_warps=num_warps,
        )
    return y


# 6. Compare values, constant rows, and non-power-of-two widths
y_triton = layernorm_triton(x, weight, bias, EPS)
torch.testing.assert_close(y_triton, y_eager, rtol=1e-5, atol=1e-5)
print("row, column | PyTorch    | Triton     | abs error")
for r, (references, results) in enumerate(zip(y_eager.cpu().tolist(), y_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:10.6f} | {got:10.6f} | {abs(ref-got):.3e}")

for rows, columns in ((0, 7), (1, 1), (3, 7), (5, 257), (17, 1003), (4, 4096)):
    values = torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    weights = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    biases = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    if rows > 0:
        values[0] = 5.0  # Constant row: normalized values are zero.
    actual = layernorm_triton(values, weights, biases, EPS)
    expected = layernorm_torch(values, weights, biases, EPS)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    if rows > 0:
        torch.testing.assert_close(actual[0], biases, rtol=0, atol=0)
print("All value, constant-row, affine, padding, and empty-batch checks passed.")


# 7. Compare eager, function, expanded, and Triton times
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("shape       eager us  function us  expanded us  Triton us  speedup  max error")
for rows, columns in ((32, 32), (256, 257), (1024, 1024), (1024, 4096)):
    values = torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    weights = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    biases = torch.randn(columns, device=DEVICE, dtype=DTYPE)
    expected = layernorm_torch(values, weights, biases, EPS)
    actual = layernorm_triton(values, weights, biases, EPS)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: F.layer_norm(values, (columns,), weight=weights, bias=biases, eps=EPS))
    function_ms = bench_ms(lambda: layernorm_torch(values, weights, biases, EPS))
    expanded_ms = bench_ms(lambda: layernorm_expanded(values, weights, biases, EPS))
    triton_ms = bench_ms(lambda: layernorm_triton(values, weights, biases, EPS))
    shape = f"{rows}x{columns}"
    print(
        f"{shape:11s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{expanded_ms*1000:12.3f} {triton_ms*1000:10.3f} "
        f"{function_ms/triton_ms:7.2f}x {error:10.3e}"
    )


# 8. Check statistics before affine scale and bias
values = torch.tensor([[1., 2., 3.], [5., 5., 5.]], device=DEVICE, dtype=DTYPE)
unit_weight = torch.ones(3, device=DEVICE, dtype=DTYPE)
zero_bias = torch.zeros(3, device=DEVICE, dtype=DTYPE)
normalized = layernorm_triton(values, unit_weight, zero_bias, EPS)
centered = values - values.mean(dim=-1, keepdim=True)
variance = centered.square().mean(dim=-1)
expected_variance = variance / (variance + EPS)
torch.testing.assert_close(
    normalized.mean(dim=-1), torch.zeros(2, device=DEVICE, dtype=DTYPE),
    rtol=0, atol=1e-6,
)
torch.testing.assert_close(
    normalized.square().mean(dim=-1), expected_variance, rtol=1e-5, atol=1e-6
)
print("Without affine parameters:\n", normalized.cpu())
print("Output mean per row:", normalized.mean(dim=-1).cpu().tolist())
print("Output variance per row:", normalized.var(dim=-1, correction=0).cpu().tolist())
```
