# 2. ReLU, SiLU, and GELU: elementwise activation kernels

Previous: [vector addition](01_vector_addition.md). Next: [fused bias + activation](03_fused_bias_activation.md).

This is a self-contained Markdown notebook. Run the examples in sections 1–8 in order in a fresh Jupyter notebook, or run the final complete code block on its own. The final block can also be saved as a Python script. The examples use contiguous, finite **FP32 vectors on one NVIDIA CUDA GPU**. They implement forward activations; the custom Triton functions do not register a PyTorch backward operation.

## The operations and their equations

An activation applies a nonlinear function independently to every input element:

$$
y_i = f(x_i), \qquad 0 \le i < N.
$$

The shape stays the same. The three choices differ in what they do to negative values and how smoothly they transition around zero.

### ReLU

The rectified linear unit keeps positive values and replaces negative values with zero:

$$
\mathrm{ReLU}(x)=\max(0,x)
=\begin{cases}0 & x\le0,\\x & x>0.\end{cases}
$$

### SiLU

The sigmoid linear unit scales the input by its logistic sigmoid:

$$
\sigma(x)=\frac{1}{1+e^{-x}}, \qquad
\mathrm{SiLU}(x)=x\sigma(x).
$$

It retains small negative outputs and changes smoothly around zero. See [PyTorch's SiLU definition](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.silu.html).

### GELU: match the approximation mode

The Gaussian error linear unit scales the input by the standard normal cumulative distribution function $\Phi$:

$$
\mathrm{GELU}(x)=x\Phi(x)
=\frac{x}{2}\left(1+\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right).
$$

This is the formula selected by `F.gelu(x, approximate="none")`. It is evaluated in floating-point arithmetic, so “exact GELU” names the formula rather than promising an infinitely precise result.

PyTorch also offers a different, tanh-based approximation:

$$
\mathrm{GELU}_{\mathrm{tanh}}(x)
\approx\frac{x}{2}\left[1+\tanh\left(\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)\right)\right].
$$

Compare that variant with `F.gelu(x, approximate="tanh")`. See [PyTorch's GELU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html).

```text
                     elementwise f(x)
input x  ─────┬────── max(0, x) ────────────────────── ReLU
              ├────── x / (1 + exp(-x)) ────────────── SiLU
              └────── x * normal-CDF(x) ────────────── GELU

Each branch preserves the input length. No neighboring values are needed.
```

The following are rounded mathematical reference values, not GPU benchmark results:

| Input | ReLU | SiLU | GELU, erf formula | GELU, tanh formula |
| ---: | ---: | ---: | ---: | ---: |
| -3 | 0.000000 | -0.142278 | -0.004050 | -0.003637 |
| -1 | 0.000000 | -0.268941 | -0.158655 | -0.158808 |
| 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 1.000000 | 0.731059 | 0.841345 | 0.841192 |
| 3 | 3.000000 | 2.857722 | 2.995950 | 2.996363 |

## 1. Imports and input

Use a CUDA-enabled PyTorch installation and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import math
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
KINDS = ("relu", "silu", "gelu", "gelu_tanh")
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

`libdevice` exposes device math functions; this notebook uses its `tanh`. The [official external-functions tutorial](https://triton-lang.org/main/getting-started/tutorials/07-extern-functions.html) explains how Triton calls these functions.

## 2. PyTorch eager: formula and built-in operation

```python
# Write the equations as ordinary eager tensor expressions.
relu_formula = torch.clamp_min(x, 0.0)
silu_formula = x / (1.0 + torch.exp(-x))
gelu_formula = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
gelu_tanh_formula = 0.5 * x * (
    1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
)

# These are the respective PyTorch built-in references.
eager_outputs = {
    "relu": F.relu(x),
    "silu": F.silu(x),
    "gelu": F.gelu(x, approximate="none"),
    "gelu_tanh": F.gelu(x, approximate="tanh"),
}
formula_outputs = {
    "relu": relu_formula,
    "silu": silu_formula,
    "gelu": gelu_formula,
    "gelu_tanh": gelu_tanh_formula,
}
for kind in KINDS:
    torch.testing.assert_close(
        formula_outputs[kind], eager_outputs[kind], rtol=1e-5, atol=1e-6
    )
    print(f"{kind:10s}:", eager_outputs[kind].cpu().tolist())
```

1. The formula expressions connect the equation to tensor code.
2. A formula such as `x / (1 + torch.exp(-x))` has multiple eager operations and intermediate tensors.
3. `F.silu(x)` uses PyTorch's implementation of the entire activation. Use the built-in as the main performance reference.
4. `.cpu().tolist()` makes the small output visible; it is not part of the timing.

## 3. PyTorch function: select an activation

```python
def activation_torch(x, kind="relu"):
    if kind == "relu":
        return F.relu(x)
    if kind == "silu":
        return F.silu(x)
    if kind == "gelu":
        return F.gelu(x, approximate="none")
    if kind == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"Unknown activation: {kind}")


for kind in KINDS:
    result = activation_torch(x, kind)
    torch.testing.assert_close(result, eager_outputs[kind], rtol=0, atol=0)
```

The string chooses the mathematical operation. This ordinary Python function still calls eager PyTorch; it is not a `torch.compile` function.

## 4. Triton math helper and GPU kernel

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
def activation_kernel(X, Y, N, KIND: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)
    result = apply_activation(values, KIND)
    tl.store(Y + offsets, result, mask=valid)
```

Step by step:

1. Compute one tile's indices, just as in vector addition.
2. Load the tile once. Math happens on the program's local values.
3. Select the activation at **compile time** using `KIND: tl.constexpr`. Calling with a different kind creates a different specialization; this is not a per-element runtime branch.
4. For SiLU, compute the exponential and scale the input. For GELU, evaluate the chosen erf or tanh formula.
5. Store each output once, masking the final partial tile.

`apply_activation` is a JIT helper used within the kernel, not a second GPU launch. Each activation therefore uses one input load and one output store per valid element. Triton's [`tl.erf`](https://triton-lang.org/main/python-api/generated/triton.language.erf.html) provides the error function for the GELU formula.

## 5. Python launch wrapper

```python
def activation_triton(x, kind="relu", block_size=256):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if kind not in KINDS:
        raise ValueError(f"Unknown activation: {kind}")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    with torch.cuda.device(x.device):
        activation_kernel[(triton.cdiv(n, block_size),)](
            x, y, n, KIND=kind, BLOCK_SIZE=block_size, num_warps=4
        )
    return y
```

The wrapper validates the layout, allocates the output, skips an empty launch, and passes the activation as a compile-time parameter. The same indexing code serves all four variants.

## 6. Compare actual values against PyTorch

```python
print("kind       input      PyTorch       Triton        abs error")
for kind in KINDS:
    expected = activation_torch(x, kind)
    actual = activation_triton(x, kind)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for value, reference, result in zip(
        x.cpu().tolist(), expected.cpu().tolist(), actual.cpu().tolist()
    ):
        print(
            f"{kind:10s} {value:7.1f} {reference:13.6f} "
            f"{result:13.6f} {abs(reference-result):13.3e}"
        )

# Include tails, an empty input, and a wider range of activation values.
for n in (0, 1, 255, 256, 257, 1000, 4097):
    values = torch.linspace(-10.0, 10.0, n, device=DEVICE, dtype=DTYPE)
    for kind in KINDS:
        torch.testing.assert_close(
            activation_triton(values, kind),
            activation_torch(values, kind),
            rtol=1e-5,
            atol=1e-6,
        )

print("All activation and boundary-size checks passed.")
```

The comparison checks

$$
|y_{\mathrm{Triton}}-y_{\mathrm{PyTorch}}|
\le \mathrm{atol}+\mathrm{rtol}\,|y_{\mathrm{PyTorch}}|.
$$

Transcendental functions and compiler arithmetic choices can produce small floating-point differences. Absolute tolerance matters near zero. The two GELU modes have intentionally different reference values; do not compare one mode to the other and attribute the difference to a kernel bug. See [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html).

The demonstrated contract is finite FP32 inputs. NaNs, infinities, extreme magnitudes, other dtypes, noncontiguous layouts, and backward gradients need separate implementation and validation if you extend the tutorial.

## 7. Compare times: built-in eager, function, and Triton

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


eager_ops = {
    "relu": lambda values: F.relu(values),
    "silu": lambda values: F.silu(values),
    "gelu": lambda values: F.gelu(values, approximate="none"),
    "gelu_tanh": lambda values: F.gelu(values, approximate="tanh"),
}

print("kind       N        eager us   function us   Triton us   speedup   max error")
for n in (1024, 65536, 1048576, 4194304):
    values = torch.randn(n, device=DEVICE, dtype=DTYPE)
    for kind in KINDS:
        expected = activation_torch(values, kind)
        actual = activation_triton(values, kind)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        error = (actual - expected).abs().max().item()

        eager_ms = bench_ms(lambda: eager_ops[kind](values))
        function_ms = bench_ms(lambda: activation_torch(values, kind))
        triton_ms = bench_ms(lambda: activation_triton(values, kind))
        print(
            f"{kind:10s} {n:8d} {eager_ms*1000:10.3f} "
            f"{function_ms*1000:13.3f} {triton_ms*1000:11.3f} "
            f"{function_ms/triton_ms:8.2f}x {error:11.3e}"
        )
```

Each timed call creates its output. The reported median GPU latency includes the behavior of the allocating API, including possible host dispatch gaps. It excludes input creation, first compilation, correctness checks, and CPU copies. [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table converts them to microseconds. It does not measure full CPU wall-clock cost.

Eager and function timings refer to the same built-in implementation. Triton is compared to that built-in, not just to a multi-operation formula. A speedup above `1.0x` indicates a faster Triton measurement; PyTorch may already perform very well. All times come from running the cell on your GPU.

## 8. See the cost of the expanded SiLU formula

```python
values = torch.randn(1048576, device=DEVICE, dtype=DTYPE)


def silu_expanded(values):
    return values / (1.0 + torch.exp(-values))


torch.testing.assert_close(
    silu_expanded(values), F.silu(values), rtol=1e-5, atol=1e-6
)
formula_ms = bench_ms(lambda: silu_expanded(values))
builtin_ms = bench_ms(lambda: F.silu(values))
triton_ms = bench_ms(lambda: activation_triton(values, "silu"))
print(f"Expanded eager formula: {formula_ms*1000:.3f} us")
print(f"PyTorch built-in SiLU:   {builtin_ms*1000:.3f} us")
print(f"Triton SiLU:            {triton_ms*1000:.3f} us")
```

This second comparison shows why expressing several eager tensor operations can be more expensive than a built-in activation or one fused kernel. The exact result depends on the GPU, input size, and software versions.

## Try next

Change the activation or tile size and rerun correctness before benchmarking. Continue to [fused bias + activation](03_fused_bias_activation.md), where the kernel combines two operations and avoids storing an intermediate tensor in global memory.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports and input
import math
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
KINDS = ("relu", "silu", "gelu", "gelu_tanh")
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: formula and built-in operation
# Write the equations as ordinary eager tensor expressions.
relu_formula = torch.clamp_min(x, 0.0)
silu_formula = x / (1.0 + torch.exp(-x))
gelu_formula = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
gelu_tanh_formula = 0.5 * x * (
    1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3))
)

# These are the respective PyTorch built-in references.
eager_outputs = {
    "relu": F.relu(x),
    "silu": F.silu(x),
    "gelu": F.gelu(x, approximate="none"),
    "gelu_tanh": F.gelu(x, approximate="tanh"),
}
formula_outputs = {
    "relu": relu_formula,
    "silu": silu_formula,
    "gelu": gelu_formula,
    "gelu_tanh": gelu_tanh_formula,
}
for kind in KINDS:
    torch.testing.assert_close(
        formula_outputs[kind], eager_outputs[kind], rtol=1e-5, atol=1e-6
    )
    print(f"{kind:10s}:", eager_outputs[kind].cpu().tolist())


# 3. PyTorch function: select an activation
def activation_torch(x, kind="relu"):
    if kind == "relu":
        return F.relu(x)
    if kind == "silu":
        return F.silu(x)
    if kind == "gelu":
        return F.gelu(x, approximate="none")
    if kind == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    raise ValueError(f"Unknown activation: {kind}")


for kind in KINDS:
    result = activation_torch(x, kind)
    torch.testing.assert_close(result, eager_outputs[kind], rtol=0, atol=0)


# 4. Triton math helper and GPU kernel
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
def activation_kernel(X, Y, N, KIND: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)
    result = apply_activation(values, KIND)
    tl.store(Y + offsets, result, mask=valid)


# 5. Python launch wrapper
def activation_triton(x, kind="relu", block_size=256):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if kind not in KINDS:
        raise ValueError(f"Unknown activation: {kind}")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    y = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return y
    with torch.cuda.device(x.device):
        activation_kernel[(triton.cdiv(n, block_size),)](
            x, y, n, KIND=kind, BLOCK_SIZE=block_size, num_warps=4
        )
    return y


# 6. Compare actual values against PyTorch
print("kind       input      PyTorch       Triton        abs error")
for kind in KINDS:
    expected = activation_torch(x, kind)
    actual = activation_triton(x, kind)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for value, reference, result in zip(
        x.cpu().tolist(), expected.cpu().tolist(), actual.cpu().tolist()
    ):
        print(
            f"{kind:10s} {value:7.1f} {reference:13.6f} "
            f"{result:13.6f} {abs(reference-result):13.3e}"
        )

# Include tails, an empty input, and a wider range of activation values.
for n in (0, 1, 255, 256, 257, 1000, 4097):
    values = torch.linspace(-10.0, 10.0, n, device=DEVICE, dtype=DTYPE)
    for kind in KINDS:
        torch.testing.assert_close(
            activation_triton(values, kind),
            activation_torch(values, kind),
            rtol=1e-5,
            atol=1e-6,
        )

print("All activation and boundary-size checks passed.")


# 7. Compare times: built-in eager, function, and Triton
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


eager_ops = {
    "relu": lambda values: F.relu(values),
    "silu": lambda values: F.silu(values),
    "gelu": lambda values: F.gelu(values, approximate="none"),
    "gelu_tanh": lambda values: F.gelu(values, approximate="tanh"),
}

print("kind       N        eager us   function us   Triton us   speedup   max error")
for n in (1024, 65536, 1048576, 4194304):
    values = torch.randn(n, device=DEVICE, dtype=DTYPE)
    for kind in KINDS:
        expected = activation_torch(values, kind)
        actual = activation_triton(values, kind)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        error = (actual - expected).abs().max().item()

        eager_ms = bench_ms(lambda: eager_ops[kind](values))
        function_ms = bench_ms(lambda: activation_torch(values, kind))
        triton_ms = bench_ms(lambda: activation_triton(values, kind))
        print(
            f"{kind:10s} {n:8d} {eager_ms*1000:10.3f} "
            f"{function_ms*1000:13.3f} {triton_ms*1000:11.3f} "
            f"{function_ms/triton_ms:8.2f}x {error:11.3e}"
        )


# 8. See the cost of the expanded SiLU formula
values = torch.randn(1048576, device=DEVICE, dtype=DTYPE)


def silu_expanded(values):
    return values / (1.0 + torch.exp(-values))


torch.testing.assert_close(
    silu_expanded(values), F.silu(values), rtol=1e-5, atol=1e-6
)
formula_ms = bench_ms(lambda: silu_expanded(values))
builtin_ms = bench_ms(lambda: F.silu(values))
triton_ms = bench_ms(lambda: activation_triton(values, "silu"))
print(f"Expanded eager formula: {formula_ms*1000:.3f} us")
print(f"PyTorch built-in SiLU:   {builtin_ms*1000:.3f} us")
print(f"Triton SiLU:            {triton_ms*1000:.3f} us")
```
