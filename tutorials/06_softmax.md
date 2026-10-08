# 6. Softmax: stable row-wise probabilities

Previous: [max reduction](05_max_reduction.md). Next: [RMSNorm](07_rmsnorm.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. The examples support contiguous **FP32 matrices of finite values on one NVIDIA CUDA GPU**, with 1–4096 columns and any nonnegative row count. They implement forward computation without a registered PyTorch backward operation.

## The operation and its equation

Softmax turns each row of scores into nonnegative values that sum to one. Here it operates over the last dimension of $X\in\mathbb{R}^{M\times N}$:

$$
y_{r,c}=\frac{e^{x_{r,c}}}{\sum_{j=0}^{N-1}e^{x_{r,j}}}.
$$

Direct exponentiation can overflow for large positive scores. Subtracting the row maximum preserves the ratio:

$$
m_r=\max_j x_{r,j},\qquad
z_{r,c}=x_{r,c}-m_r,\qquad
y_{r,c}=\frac{e^{z_{r,c}}}{\sum_j e^{z_{r,j}}}.
$$

Every valid shifted score is nonpositive, and at least one is zero. This keeps the largest exponential at one. See the [PyTorch softmax reference](https://docs.pytorch.org/docs/stable/generated/torch.softmax.html).

```text
row [1, 2, 3]      -> subtract max 3    -> [-2, -1, 0]
                                       -> [0.135335, 0.367879, 1]
                                       -> divide by sum 1.503215
                                       -> [0.090031, 0.244728, 0.665241]

row [1001,1002,1003] -> subtract max 1003 -> the same shifted row
                                       -> the same probabilities
```

```mermaid
flowchart LR
    X[Load one row] --> M[Row maximum]
    X --> S[Subtract maximum]
    M --> S
    S --> E[Exponentiate]
    E --> D[Sum exponentials]
    E --> Q[Divide by sum]
    D --> Q
    Q --> Y[Store one row]
```

The Triton kernel fuses this chain into one program per row. It retains the loaded row locally instead of storing each intermediate in a separate tensor. This teaching pattern is also used in the [official fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).

## 1. Imports and example scores

Use a CUDA-enabled PyTorch build and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import torch
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")

DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 2., 3.], [1001., 1002., 1003.]], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

## 2. PyTorch eager: expand the stable equation

```python
row_max = x.amax(dim=-1, keepdim=True)
shifted = x - row_max
numerator = torch.exp(shifted)
denominator = numerator.sum(dim=-1, keepdim=True)
y_formula = numerator / denominator
y_eager = torch.softmax(x, dim=-1)
torch.testing.assert_close(y_formula, y_eager, rtol=1e-5, atol=1e-6)
print("Shifted scores:\n", shifted.cpu())
print("Softmax:\n", y_eager.cpu())
```

1. Reduce along the column dimension, retaining shape `(M, 1)` for broadcasting.
2. Subtract the row maximum, then exponentiate the shifted scores.
3. Sum each row of exponentials and divide each element by its row's sum.
4. Compare with PyTorch's built-in softmax, which is the primary performance reference.

Both example rows produce approximately `[0.090031, 0.244728, 0.665241]`.

## 3. PyTorch functions: built-in and expanded references

```python
def softmax_torch(x):
    return torch.softmax(x, dim=-1)


def softmax_expanded(x):
    shifted = x - x.amax(dim=-1, keepdim=True)
    numerator = torch.exp(shifted)
    return numerator / numerator.sum(dim=-1, keepdim=True)


torch.testing.assert_close(softmax_torch(x), y_eager, rtol=0, atol=0)
torch.testing.assert_close(softmax_expanded(x), y_eager, rtol=1e-5, atol=1e-6)
```

These functions still execute eagerly. Timing both makes the cost of the expanded formula visible while retaining the optimized built-in as the main comparison.

## 4. Triton: load, reduce, normalize, and store one row

```python
@triton.jit
def softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS

    # Contiguous rows start at row * N_COLS.
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=-float("inf"))
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(Y + row * N_COLS + columns, result, mask=valid)
```

1. A program ID selects a row, rather than a tile of a flattened full matrix.
2. Pad the logical row to a power-of-two tile width. Invalid positions load negative infinity.
3. Reduce the row maximum, subtract it, and exponentiate. Padded positions become `exp(-inf) = 0`.
4. Sum the exponentials and divide by the sum.
5. Store only the actual columns.

The same padding value supports both reductions: it cannot win the maximum and contributes zero after exponentiation. The positive column count ensures at least one valid score exists.

## 5. Python wrapper: choose the padded row width

```python
def softmax_triton(x):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous matrix.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    rows, columns = x.shape
    if not 1 <= columns <= 4096:
        raise ValueError("This teaching kernel supports 1–4096 columns.")
    y = torch.empty_like(x)
    if rows == 0:
        return y
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size <= 1024 else 8
    with torch.cuda.device(x.device):
        softmax_kernel[(rows,)](
            x, y, N_COLS=columns, BLOCK_SIZE=block_size, num_warps=num_warps
        )
    return y
```

For 1003 columns, `block_size=1024`; the remaining 21 positions are masked. The width limit bounds local storage in this introductory implementation. It is not a universal softmax or hardware limit. Wider rows need a different kernel strategy or additional tuning.

## 6. Compare values, row sums, and boundary shapes

```python
y_triton = softmax_triton(x)
torch.testing.assert_close(y_triton, y_eager, rtol=1e-5, atol=1e-6)
print("row, column | PyTorch  | Triton   | abs error")
for r, (references, results) in enumerate(zip(y_eager.cpu().tolist(), y_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:.6f} | {got:.6f} | {abs(ref-got):.3e}")

for rows, columns in ((0, 7), (1, 1), (3, 7), (5, 257), (17, 1003), (4, 4096)):
    values = 5.0 * torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    actual = softmax_triton(values)
    expected = softmax_torch(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(rows, device=DEVICE, dtype=DTYPE),
        rtol=1e-5, atol=1e-6,
    )
    assert bool(torch.all(actual >= 0))
print("All value, probability, padding, and empty-row checks passed.")
```

The shifted example checks stability on large scores. Single-column rows should contain `1`. Non-power-of-two widths test padding. Small differences are expected because exponentials and reduction order are evaluated in floating-point arithmetic. Rows containing NaNs or infinities require an explicit policy beyond this finite-input example.

## 7. Compare built-in, function, expanded, and Triton times

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
    expected, actual = softmax_torch(values), softmax_triton(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: torch.softmax(values, dim=-1))
    function_ms = bench_ms(lambda: softmax_torch(values))
    expanded_ms = bench_ms(lambda: softmax_expanded(values))
    triton_ms = bench_ms(lambda: softmax_triton(values))
    shape = f"{rows}x{columns}"
    print(
        f"{shape:11s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{expanded_ms*1000:12.3f} {triton_ms*1000:10.3f} "
        f"{function_ms/triton_ms:7.2f}x {error:10.3e}"
    )
```

Each callable allocates its output; the expanded formula also creates intermediates. The speedup uses the built-in PyTorch function time, not the expanded formula time. The built-in already has an optimized implementation, so fusion alone does not guarantee Triton will win.

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table reports median GPU latency in microseconds. It may include host dispatch gaps. Input creation, first compilation, assertions, CPU copies, and printing are outside timing. Full CPU wall-clock time is a separate measurement. Run the block to obtain actual timings on your GPU.

## 8. Compare preallocated output launches

```python
values = torch.randn(1024, 1003, device=DEVICE, dtype=DTYPE)
torch_out, triton_out = torch.empty_like(values), torch.empty_like(values)
rows, columns = values.shape
block_size = triton.next_power_of_2(columns)


def launch_torch_softmax():
    torch.softmax(values, dim=-1, out=torch_out)


def launch_triton_softmax():
    softmax_kernel[(rows,)](
        values, triton_out, N_COLS=columns, BLOCK_SIZE=block_size, num_warps=4
    )


launch_torch_softmax()
launch_triton_softmax()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-5, atol=1e-6)
torch_ms, triton_ms = bench_ms(launch_torch_softmax), bench_ms(launch_triton_softmax)
print(f"Preallocated PyTorch: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
```

Both outputs are allocated before timing. The inputs have `requires_grad=False`, as required for these `out=` examples. This is a distinct comparison from the allocating wrappers in section 7.

## Try next

Continue with [RMSNorm](07_rmsnorm.md), which uses a row reduction to rescale features rather than produce probabilities.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports and example scores
import torch
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")

DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
x = torch.tensor([[1., 2., 3.], [1001., 1002., 1003.]], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: expand the stable equation
row_max = x.amax(dim=-1, keepdim=True)
shifted = x - row_max
numerator = torch.exp(shifted)
denominator = numerator.sum(dim=-1, keepdim=True)
y_formula = numerator / denominator
y_eager = torch.softmax(x, dim=-1)
torch.testing.assert_close(y_formula, y_eager, rtol=1e-5, atol=1e-6)
print("Shifted scores:\n", shifted.cpu())
print("Softmax:\n", y_eager.cpu())


# 3. PyTorch functions: built-in and expanded references
def softmax_torch(x):
    return torch.softmax(x, dim=-1)


def softmax_expanded(x):
    shifted = x - x.amax(dim=-1, keepdim=True)
    numerator = torch.exp(shifted)
    return numerator / numerator.sum(dim=-1, keepdim=True)


torch.testing.assert_close(softmax_torch(x), y_eager, rtol=0, atol=0)
torch.testing.assert_close(softmax_expanded(x), y_eager, rtol=1e-5, atol=1e-6)


# 4. Triton: load, reduce, normalize, and store one row
@triton.jit
def softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    valid = columns < N_COLS

    # Contiguous rows start at row * N_COLS.
    values = tl.load(X + row * N_COLS + columns, mask=valid, other=-float("inf"))
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator
    tl.store(Y + row * N_COLS + columns, result, mask=valid)


# 5. Python wrapper: choose the padded row width
def softmax_triton(x):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous matrix.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    rows, columns = x.shape
    if not 1 <= columns <= 4096:
        raise ValueError("This teaching kernel supports 1–4096 columns.")
    y = torch.empty_like(x)
    if rows == 0:
        return y
    block_size = triton.next_power_of_2(columns)
    num_warps = 4 if block_size <= 1024 else 8
    with torch.cuda.device(x.device):
        softmax_kernel[(rows,)](
            x, y, N_COLS=columns, BLOCK_SIZE=block_size, num_warps=num_warps
        )
    return y


# 6. Compare values, row sums, and boundary shapes
y_triton = softmax_triton(x)
torch.testing.assert_close(y_triton, y_eager, rtol=1e-5, atol=1e-6)
print("row, column | PyTorch  | Triton   | abs error")
for r, (references, results) in enumerate(zip(y_eager.cpu().tolist(), y_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:.6f} | {got:.6f} | {abs(ref-got):.3e}")

for rows, columns in ((0, 7), (1, 1), (3, 7), (5, 257), (17, 1003), (4, 4096)):
    values = 5.0 * torch.randn(rows, columns, device=DEVICE, dtype=DTYPE)
    actual = softmax_triton(values)
    expected = softmax_torch(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        actual.sum(dim=-1), torch.ones(rows, device=DEVICE, dtype=DTYPE),
        rtol=1e-5, atol=1e-6,
    )
    assert bool(torch.all(actual >= 0))
print("All value, probability, padding, and empty-row checks passed.")


# 7. Compare built-in, function, expanded, and Triton times
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
    expected, actual = softmax_torch(values), softmax_triton(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: torch.softmax(values, dim=-1))
    function_ms = bench_ms(lambda: softmax_torch(values))
    expanded_ms = bench_ms(lambda: softmax_expanded(values))
    triton_ms = bench_ms(lambda: softmax_triton(values))
    shape = f"{rows}x{columns}"
    print(
        f"{shape:11s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{expanded_ms*1000:12.3f} {triton_ms*1000:10.3f} "
        f"{function_ms/triton_ms:7.2f}x {error:10.3e}"
    )


# 8. Compare preallocated output launches
values = torch.randn(1024, 1003, device=DEVICE, dtype=DTYPE)
torch_out, triton_out = torch.empty_like(values), torch.empty_like(values)
rows, columns = values.shape
block_size = triton.next_power_of_2(columns)


def launch_torch_softmax():
    torch.softmax(values, dim=-1, out=torch_out)


def launch_triton_softmax():
    softmax_kernel[(rows,)](
        values, triton_out, N_COLS=columns, BLOCK_SIZE=block_size, num_warps=4
    )


launch_torch_softmax()
launch_triton_softmax()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-5, atol=1e-6)
torch_ms, triton_ms = bench_ms(launch_torch_softmax), bench_ms(launch_triton_softmax)
print(f"Preallocated PyTorch: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
```
