# 10. MatMul + bias + activation: fuse the output epilogue

Previous: [MatMul](09_matmul.md). Next: [batched MatMul](11_batched_matmul.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous, finite FP32 CUDA tensors on one NVIDIA GPU; products and sums must remain representable in FP32. The examples implement forward computation without a registered backward operation.

## Concepts, shapes, and equations

A dense layer often multiplies an input by a weight matrix, adds a feature bias, and applies an activation. Let $A\in\mathbb{R}^{M\times K}$, $B\in\mathbb{R}^{K\times N}$, and $b\in\mathbb{R}^{N}$:

$$
z_{m,n}=\sum_{k=0}^{K-1}A_{m,k}B_{k,n}+b_n,\qquad y_{m,n}=f(z_{m,n}).
$$

The bias is broadcast across rows. The activation acts independently on each completed dot product. This guide implements ReLU, SiLU, and both GELU formulas from [the activation guide](02_relu_silu_gelu.md).

$$
\operatorname{ReLU}(z)=\max(0,z),\quad
\operatorname{SiLU}(z)=z/(1+e^{-z}),\quad
\operatorname{GELU}(z)=\tfrac12z\left(1+\operatorname{erf}(z/\sqrt2)\right).
$$

The tanh GELU variant uses `F.gelu(..., approximate="tanh")`; the erf variant uses `approximate="none"`. Matching that mode is part of matching the operation.

### What is an epilogue?

The matmul loop accumulates an output tile over K. Its **epilogue** is the work performed after the final accumulation and before writing the tile. Bias and activation fit here because they need only the completed accumulator and a short bias vector.

```text
A/B tiles -> repeated dot products -> completed FP32 accumulator
                                             |
                                     add column bias
                                             |
                                     apply activation
                                             |
                                     one output store
```

In eager PyTorch, `A @ B`, bias addition, and activation create separate results and device operations. Fusing the epilogue avoids storing and rereading the full raw matmul result and the full biased result. It does not remove the matmul's arithmetic or guarantee a faster result than a tuned library.

### Apply activation after the full sum

In general, $f(u+v)\ne f(u)+f(v)$. Applying ReLU inside the K loop changes the equation. For partial contributions `-3` and `2`, `ReLU(-3+2)=0`, while `ReLU(-3)+ReLU(2)=2`. The code therefore adds the bias and activates exactly once after the K loop.

For an empty shared dimension `K=0`, the dot product is zero but the output is $f(b_n)$, broadcast across rows. It must not be replaced by an all-zero output.

## 1. Imports and example inputs

Use a CUDA-enabled PyTorch build and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from triton.language.extra import libdevice

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
# Match the Triton kernels' explicit FP32 IEEE dot products.
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
KINDS = ("relu", "silu", "gelu", "gelu_tanh")
a = torch.tensor([[1., -2.], [3., 1.]], device=DEVICE, dtype=DTYPE)
b = torch.tensor([[2., -1.], [1., 3.]], device=DEVICE, dtype=DTYPE)
bias = torch.tensor([-1., 2.], device=DEVICE, dtype=DTYPE)
```

Use the same FP32 precision setting as the standalone MatMul guide.

## 2. PyTorch eager: multiply, bias, activate

Write the three steps explicitly.

```python
raw = a @ b
biased = raw + bias
y_eager = F.relu(biased)
print("MatMul:\n", raw.cpu())
print("With bias:\n", biased.cpu())
print("ReLU output:\n", y_eager.cpu())
```

The raw result is `[[0,-7],[7,0]]`, the biased result is `[[-1,-5],[6,2]]`, and the ReLU output is `[[0,0],[6,2]]`.

## 3. PyTorch function: choose the activation

This function still executes the operations eagerly.

```python
def matmul_bias_activation_torch(a, b, bias, kind="relu"):
    value = a @ b + bias
    if kind == "relu":
        return F.relu(value)
    if kind == "silu":
        return F.silu(value)
    if kind == "gelu":
        return F.gelu(value, approximate="none")
    if kind == "gelu_tanh":
        return F.gelu(value, approximate="tanh")
    raise ValueError(f"Unknown activation: {kind}")


torch.testing.assert_close(matmul_bias_activation_torch(a,b,bias), y_eager, rtol=0, atol=0)
```

The function is also the benchmark reference for each activation.

## 4. Triton kernel: matmul loop followed by a fused epilogue

The tile indexing and K loop are the same as [MatMul](09_matmul.md).

```python
@triton.jit
def apply_activation(value, KIND: tl.constexpr):
    if KIND == "relu":
        result = tl.maximum(value, 0.0)
    elif KIND == "silu":
        result = value / (1.0 + tl.exp(-value))
    elif KIND == "gelu":
        result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    else:  # The Python wrapper permits only "gelu_tanh" here.
        inner = 0.7978845608028654 * (
            value + 0.044715 * value * value * value
        )
        result = 0.5 * value * (1.0 + libdevice.tanh(inner))
    return result


@triton.jit
def matmul_bias_activation_kernel(
    A, B, Bias, Y, M, N, K, KIND: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    # Epilogue: add each column's bias and activate before the only store.
    bias_tile = tl.load(Bias + columns, mask=columns < N, other=0.0)
    result = apply_activation(accumulator + bias_tile[None, :], KIND)
    c_ptrs = Y + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, result, mask=c_valid)
```

1. Accumulate the entire dot product in FP32.
2. Load one bias per output column and broadcast it across the tile rows.
3. Apply a compile-time-selected activation to the complete biased tile.
4. Store only valid output positions.

The JIT activation helper is part of this kernel, not another launch.

## 5. Python wrapper: validate and launch

The kernel also handles `K=0`: its zero accumulator proceeds through the epilogue.

```python
def matmul_bias_activation_triton(a, b, bias, kind="relu"):
    if not all(t.is_cuda and t.device == a.device for t in (a,b,bias)):
        raise ValueError("Inputs must be CUDA tensors on one device.")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected matrices with matching K dimensions.")
    if bias.ndim != 1 or bias.numel() != b.shape[1]:
        raise ValueError("Bias must have one value per output column.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (a,b,bias)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    if kind not in KINDS:
        raise ValueError(f"Unknown activation: {kind}")
    m, k = a.shape
    n = b.shape[1]
    y = torch.empty((m,n), device=a.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return y
    with torch.cuda.device(a.device):
        matmul_bias_activation_kernel[(triton.cdiv(m,32),triton.cdiv(n,32))](
            a,b,bias,y,m,n,k,KIND=kind,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return y
```

The bias is not expanded into a full matrix. All three pointer layouts assume contiguous storage.

## 6. Match values, activations, and tile boundaries

Print actual values and check a zero-K case as well as partial tiles.

```python
for kind in KINDS:
    expected = matmul_bias_activation_torch(a,b,bias,kind)
    actual = matmul_bias_activation_triton(a,b,bias,kind)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    print(kind, "PyTorch:", expected.cpu().tolist(), "Triton:", actual.cpu().tolist())
    print("Max error:", (actual-expected).abs().max().item())
for m,n,k in ((0,7,3),(3,0,7),(3,5,0),(1,1,1),(17,19,23),(33,65,31)):
    left = torch.randn(m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(k,n,device=DEVICE,dtype=DTYPE)
    bias_values = torch.randn(n,device=DEVICE,dtype=DTYPE)
    for kind in KINDS:
        torch.testing.assert_close(
            matmul_bias_activation_triton(left,right,bias_values,kind),
            matmul_bias_activation_torch(left,right,bias_values,kind),rtol=1e-4,atol=1e-3,
        )
print("All fused-epilogue checks passed.")
```

FP32 accumulation order and activation math can differ slightly. Inspect absolute errors, especially near zero, rather than expecting bitwise agreement for every random dot product.

## 7. Compare eager, function, and Triton times

Use the same inputs and activation formula for every path.

```python
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

eager_ops = {
    "relu": lambda a,b,c: F.relu(a @ b + c),
    "silu": lambda a,b,c: F.silu(a @ b + c),
    "gelu": lambda a,b,c: F.gelu(a @ b + c, approximate="none"),
    "gelu_tanh": lambda a,b,c: F.gelu(a @ b + c, approximate="tanh"),
}
print("kind      M,N,K       eager us  function us  Triton us  speedup  max error")
for m,n,k in ((64,64,64),(128,256,64),(256,256,256)):
    left = torch.randn(m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(k,n,device=DEVICE,dtype=DTYPE)
    bias_values = torch.randn(n,device=DEVICE,dtype=DTYPE)
    for kind in KINDS:
        expected = matmul_bias_activation_torch(left,right,bias_values,kind)
        actual = matmul_bias_activation_triton(left,right,bias_values,kind)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
        error = (actual-expected).abs().max().item()
        e = bench_ms(lambda: eager_ops[kind](left,right,bias_values))
        p = bench_ms(lambda: matmul_bias_activation_torch(left,right,bias_values,kind))
        t = bench_ms(lambda: matmul_bias_activation_triton(left,right,bias_values,kind))
        print(f"{kind:9s} {m},{n},{k} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

Each timed call allocates its output. Input creation, first compilation, assertions, CPU copies, and printing are outside timing. [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock time. A speedup above `1.0x` means the Triton measurement was faster. Run the code for actual timings; no speedup is assumed.

## 8. Check why activation cannot move inside the K loop

A small eager example demonstrates the order dependency.

```python
partials = torch.tensor([-3.,2.],device=DEVICE,dtype=DTYPE)
correct = F.relu(partials.sum())
wrong = F.relu(partials).sum()
print("Activation after full sum:", correct.item())
print("Sum of activated partials:", wrong.item())
assert correct.item() == 0.0 and wrong.item() == 2.0
```

The fused kernel preserves the mathematical order while avoiding intermediate global-memory stores.

## Try next

Continue with [batched MatMul](11_batched_matmul.md), where an additional program ID chooses which independent matrix pair to multiply.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code: imports, references, kernels, launch helpers, correctness checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and example inputs
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing
from triton.language.extra import libdevice

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
DTYPE = torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
# Match the Triton kernels' explicit FP32 IEEE dot products.
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
KINDS = ("relu", "silu", "gelu", "gelu_tanh")
a = torch.tensor([[1., -2.], [3., 1.]], device=DEVICE, dtype=DTYPE)
b = torch.tensor([[2., -1.], [1., 3.]], device=DEVICE, dtype=DTYPE)
bias = torch.tensor([-1., 2.], device=DEVICE, dtype=DTYPE)


# 2. PyTorch eager: multiply, bias, activate
raw = a @ b
biased = raw + bias
y_eager = F.relu(biased)
print("MatMul:\n", raw.cpu())
print("With bias:\n", biased.cpu())
print("ReLU output:\n", y_eager.cpu())


# 3. PyTorch function: choose the activation
def matmul_bias_activation_torch(a, b, bias, kind="relu"):
    value = a @ b + bias
    if kind == "relu":
        return F.relu(value)
    if kind == "silu":
        return F.silu(value)
    if kind == "gelu":
        return F.gelu(value, approximate="none")
    if kind == "gelu_tanh":
        return F.gelu(value, approximate="tanh")
    raise ValueError(f"Unknown activation: {kind}")


torch.testing.assert_close(matmul_bias_activation_torch(a,b,bias), y_eager, rtol=0, atol=0)


# 4. Triton kernel: matmul loop followed by a fused epilogue
@triton.jit
def apply_activation(value, KIND: tl.constexpr):
    if KIND == "relu":
        result = tl.maximum(value, 0.0)
    elif KIND == "silu":
        result = value / (1.0 + tl.exp(-value))
    elif KIND == "gelu":
        result = 0.5 * value * (1.0 + tl.erf(value * 0.7071067811865476))
    else:  # The Python wrapper permits only "gelu_tanh" here.
        inner = 0.7978845608028654 * (
            value + 0.044715 * value * value * value
        )
        result = 0.5 * value * (1.0 + libdevice.tanh(inner))
    return result


@triton.jit
def matmul_bias_activation_kernel(
    A, B, Bias, Y, M, N, K, KIND: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    # Epilogue: add each column's bias and activate before the only store.
    bias_tile = tl.load(Bias + columns, mask=columns < N, other=0.0)
    result = apply_activation(accumulator + bias_tile[None, :], KIND)
    c_ptrs = Y + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, result, mask=c_valid)


# 5. Python wrapper: validate and launch
def matmul_bias_activation_triton(a, b, bias, kind="relu"):
    if not all(t.is_cuda and t.device == a.device for t in (a,b,bias)):
        raise ValueError("Inputs must be CUDA tensors on one device.")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected matrices with matching K dimensions.")
    if bias.ndim != 1 or bias.numel() != b.shape[1]:
        raise ValueError("Bias must have one value per output column.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (a,b,bias)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    if kind not in KINDS:
        raise ValueError(f"Unknown activation: {kind}")
    m, k = a.shape
    n = b.shape[1]
    y = torch.empty((m,n), device=a.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return y
    with torch.cuda.device(a.device):
        matmul_bias_activation_kernel[(triton.cdiv(m,32),triton.cdiv(n,32))](
            a,b,bias,y,m,n,k,KIND=kind,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return y


# 6. Match values, activations, and tile boundaries
for kind in KINDS:
    expected = matmul_bias_activation_torch(a,b,bias,kind)
    actual = matmul_bias_activation_triton(a,b,bias,kind)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    print(kind, "PyTorch:", expected.cpu().tolist(), "Triton:", actual.cpu().tolist())
    print("Max error:", (actual-expected).abs().max().item())
for m,n,k in ((0,7,3),(3,0,7),(3,5,0),(1,1,1),(17,19,23),(33,65,31)):
    left = torch.randn(m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(k,n,device=DEVICE,dtype=DTYPE)
    bias_values = torch.randn(n,device=DEVICE,dtype=DTYPE)
    for kind in KINDS:
        torch.testing.assert_close(
            matmul_bias_activation_triton(left,right,bias_values,kind),
            matmul_bias_activation_torch(left,right,bias_values,kind),rtol=1e-4,atol=1e-3,
        )
print("All fused-epilogue checks passed.")


# 7. Compare eager, function, and Triton times
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

eager_ops = {
    "relu": lambda a,b,c: F.relu(a @ b + c),
    "silu": lambda a,b,c: F.silu(a @ b + c),
    "gelu": lambda a,b,c: F.gelu(a @ b + c, approximate="none"),
    "gelu_tanh": lambda a,b,c: F.gelu(a @ b + c, approximate="tanh"),
}
print("kind      M,N,K       eager us  function us  Triton us  speedup  max error")
for m,n,k in ((64,64,64),(128,256,64),(256,256,256)):
    left = torch.randn(m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(k,n,device=DEVICE,dtype=DTYPE)
    bias_values = torch.randn(n,device=DEVICE,dtype=DTYPE)
    for kind in KINDS:
        expected = matmul_bias_activation_torch(left,right,bias_values,kind)
        actual = matmul_bias_activation_triton(left,right,bias_values,kind)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
        error = (actual-expected).abs().max().item()
        e = bench_ms(lambda: eager_ops[kind](left,right,bias_values))
        p = bench_ms(lambda: matmul_bias_activation_torch(left,right,bias_values,kind))
        t = bench_ms(lambda: matmul_bias_activation_triton(left,right,bias_values,kind))
        print(f"{kind:9s} {m},{n},{k} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Check why activation cannot move inside the K loop
partials = torch.tensor([-3.,2.],device=DEVICE,dtype=DTYPE)
correct = F.relu(partials.sum())
wrong = F.relu(partials).sum()
print("Activation after full sum:", correct.item())
print("Sum of activated partials:", wrong.item())
assert correct.item() == 0.0 and wrong.item() == 2.0
```
