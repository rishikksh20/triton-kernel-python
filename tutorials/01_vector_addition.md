# 1. Vector addition: from PyTorch to a Triton kernel

Start here, then continue with [activations](02_relu_silu_gelu.md), [fused bias + activation](03_fused_bias_activation.md), and [sum reduction](04_sum_reduction.md). For the GPU execution model, see [BASICS.md](../BASICS.md).

This is a Markdown notebook: run the examples in sections 1–8 in order in a fresh notebook, or run the final complete code block on its own. Every guide includes its own imports and definitions. The final block can also be saved as a `.py` file. The examples use contiguous, finite, **FP32 tensors on one NVIDIA CUDA GPU** and implement forward computation. Custom Triton kernels here do not register a PyTorch backward operation.

## The operation and its equation

Vector addition adds corresponding elements of two vectors. Inputs and output have the same length; no value depends on a neighboring element.

For $a,b\in\mathbb{R}^{N}$:

$$
c_i = a_i + b_i, \qquad 0 \le i < N.
$$

For example:

```text
index       0      1      2      3      4
a        [ 1.0,  -2.0,   3.5,   0.0,   8.0 ]
b        [ 4.0,   5.0,  -1.5,   7.0,  -3.0 ]
           +      +      +      +      +
c        [ 5.0,   3.0,   2.0,   7.0,   5.0 ]
```

There are $N$ additions. The logical memory traffic is two input reads and one output write, or $3N\times4=12N$ bytes for FP32. Large vector additions often spend more time moving data than doing arithmetic.

## 1. Set up the notebook

Use an environment with a CUDA-enabled PyTorch build and Triton. Install PyTorch using the command appropriate to your CUDA setup from the [official installation selector](https://pytorch.org/get-started/locally/); see the [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html) for Triton. If these are already installed, run this cell directly.

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

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

`torch` creates and owns the tensors. Triton receives pointers to their GPU storage. `triton.language` supplies operations used inside a GPU kernel, while `triton.testing` supplies the timing helper.

## 2. PyTorch eager: write the operation directly

```python
a = torch.tensor([1.0, -2.0, 3.5, 0.0, 8.0], device=DEVICE, dtype=DTYPE)
b = torch.tensor([4.0, 5.0, -1.5, 7.0, -3.0], device=DEVICE, dtype=DTYPE)

c_eager = a + b
print("a:      ", a.cpu().tolist())
print("b:      ", b.cpu().tolist())
print("a + b:  ", c_eager.cpu().tolist())
```

1. `torch.tensor(..., device=DEVICE)` copies the small example inputs to GPU memory.
2. `a + b` invokes PyTorch's eager addition operation and creates an output tensor.
3. `.cpu().tolist()` brings the result back so you can inspect the values. Keep these copies outside benchmarks.

The expected output is `[5.0, 3.0, 2.0, 7.0, 5.0]`.

## 3. PyTorch function: make it reusable

```python
def vector_add_torch(a, b):
    return torch.add(a, b)


c_function = vector_add_torch(a, b)
torch.testing.assert_close(c_function, c_eager, rtol=0, atol=0)
print("Function:", c_function.cpu().tolist())
```

This is an ordinary Python function. Its body still executes eagerly; defining a function does not compile or fuse its operations. For these equal-shaped vectors, `torch.add(a, b)` and `a + b` express the same calculation.

## How a Triton program owns a tile

The kernel divides the vector into tiles. A **program instance** processes one tile; a program ID is not an individual CUDA thread ID.

```text
N = 10, BLOCK_SIZE = 4                 (small illustration)

program 0     offsets [0, 1, 2, 3]     all valid
program 1     offsets [4, 5, 6, 7]     all valid
program 2     offsets [8, 9,10,11]     mask [True, True, False, False]
                          ^  ^
                      no load or store
```

The actual examples use larger power-of-two tile sizes. A mask protects the last tile when the vector length is not a multiple of the tile size. This is the indexing pattern used in the [official Triton vector-add tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html).

## 4. Write the Triton GPU kernel

```python
@triton.jit
def vector_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    # 1. Find the tile owned by this program.
    pid = tl.program_id(axis=0)

    # 2. Compute the logical element indices in that tile.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 3. Load both inputs. Invalid positions receive zero locally.
    a_tile = tl.load(A + offsets, mask=valid, other=0.0)
    b_tile = tl.load(B + offsets, mask=valid, other=0.0)

    # 4. Add the tiles element by element.
    c_tile = a_tile + b_tile

    # 5. Write only real output elements.
    tl.store(C + offsets, c_tile, mask=valid)
```

| Line or concept | Why it is needed |
| --- | --- |
| `@triton.jit` | Marks the function for Triton's GPU compiler |
| `BLOCK_SIZE: tl.constexpr` | Makes the tile shape known at compilation time |
| `tl.program_id(0)` | Gives each program a distinct output region |
| `tl.arange(0, BLOCK_SIZE)` | Creates the logical indices within the tile |
| `A + offsets` | Computes addresses relative to the input pointer |
| `mask=valid` | Prevents access beyond the vector's storage |
| `other=0.0` | Defines the local value at masked input positions |
| `tl.store(...)` | Writes the result back to the PyTorch output tensor |

The tile is a logical tensor distributed across the program's execution resources. `BLOCK_SIZE=256` does not mean “256 CUDA threads”; `num_warps` controls a different aspect of the launch.

## 5. Allocate the output and launch from Python

```python
def vector_add_triton(a, b, block_size=256):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("Inputs must be vectors with the same shape.")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 inputs.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("Inputs must be contiguous.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    c = torch.empty_like(a)
    n = a.numel()
    if n == 0:
        return c

    grid = (triton.cdiv(n, block_size),)
    with torch.cuda.device(a.device):
        vector_add_kernel[grid](
            a, b, c, n, BLOCK_SIZE=block_size, num_warps=4
        )
    return c
```

1. Validate the layout that the pointer arithmetic assumes.
2. Allocate output storage; the GPU kernel itself does not allocate a PyTorch tensor.
3. Use ceiling division to launch enough programs for every element.
4. Launch on the input device. The first launch compiles this specialization.
5. Return the output tensor. Work is asynchronous; PyTorch operations on the same CUDA stream observe the correct ordering.

An empty input returns an empty output without launching a zero-sized grid.

## 6. Match output values and check the tail

```python
c_triton = vector_add_triton(a, b)
torch.testing.assert_close(c_triton, c_eager, rtol=0, atol=0)

print("index | eager     | function  | Triton    | abs error")
for i, (e, f, t) in enumerate(zip(
    c_eager.cpu().tolist(),
    c_function.cpu().tolist(),
    c_triton.cpu().tolist(),
)):
    print(f"{i:5d} | {e:9.6f} | {f:9.6f} | {t:9.6f} | {abs(e-t):.3e}")

for n in (0, 1, 127, 128, 255, 256, 257, 1000, 4097):
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    z = torch.randn_like(x)
    expected = vector_add_torch(x, z)
    for block_size in (128, 256, 512, 1024):
        actual = vector_add_triton(x, z, block_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

print("All boundary-size and block-size checks passed.")
```

These finite FP32 additions use the same per-element operation, so the example checks exact equality. Sizes just below and above a tile boundary exercise the mask rather than only complete tiles. All five hand-worked values should have zero absolute error.

## 7. Compare execution times with the same inputs

CUDA work is asynchronous. A plain `time.time()` around a launch can finish before the GPU does. Triton's [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) measures GPU execution and returns milliseconds. Here we request its median and print microseconds.

```python
def bench_ms(fn):
    # Compile and warm up before measuring steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us   function us   Triton us   speedup   max abs error")
for n in (1024, 65536, 1048576, 4194304):
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    z = torch.randn_like(x)
    expected = vector_add_torch(x, z)
    actual = vector_add_triton(x, z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    max_error = (actual - expected).abs().max().item()

    eager_ms = bench_ms(lambda: x + z)
    function_ms = bench_ms(lambda: vector_add_torch(x, z))
    triton_ms = bench_ms(lambda: vector_add_triton(x, z))
    speedup = function_ms / triton_ms
    print(
        f"{n:8d} {eager_ms*1000:10.3f} {function_ms*1000:13.3f} "
        f"{triton_ms*1000:11.3f} {speedup:9.2f}x {max_error:14.3e}"
    )
```

All three callables create an output during each call. This measures the steady-state GPU latency of these allocating APIs, including any gaps caused by host dispatch and allocation; it is not a full CPU wall-clock measurement. Inputs, compilation, correctness assertions, printing, and CPU copies are outside the measured region.

The function and direct eager form should have similar timings, subject to measurement noise. A speedup above `1.0x` means Triton is faster in that measurement. Small inputs often expose launch overhead, while larger inputs expose memory throughput. No timing numbers are pre-filled: this cell produces measurements for your GPU and installed versions.

## 8. Compare launches with preallocated output

Reuse output storage to isolate the launch and device work more closely. Both implementations now receive an output buffer allocated before timing.

```python
n = 4194304
x = torch.randn(n, device=DEVICE, dtype=DTYPE)
z = torch.randn_like(x)
torch_out = torch.empty_like(x)
triton_out = torch.empty_like(x)
block_size = 256
grid = (triton.cdiv(n, block_size),)


def launch_torch_add():
    torch.add(x, z, out=torch_out)


def launch_triton_add():
    vector_add_kernel[grid](
        x, z, triton_out, n, BLOCK_SIZE=block_size, num_warps=4
    )


launch_torch_add()
launch_triton_add()
torch.testing.assert_close(triton_out, torch_out, rtol=0, atol=0)

torch_ms = bench_ms(launch_torch_add)
triton_ms = bench_ms(launch_triton_add)
logical_bytes = 3 * n * x.element_size()
print(f"PyTorch: {torch_ms*1000:.3f} us")
print(f"Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Triton effective bandwidth: {logical_bytes/(triton_ms*1e-3)/1e9:.1f} GB/s")
```

The bandwidth uses logical bytes, not a measurement of physical HBM traffic. Caches and the benchmark's execution pattern affect the result. Record the GPU, package versions, length, dtype, tile size, and timing method when sharing measurements.

## Try next

Change `block_size` and repeat section 8 after updating its grid. Then move to [ReLU / SiLU / GELU](02_relu_silu_gelu.md): the same load–compute–store skeleton will implement richer elementwise math.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Set up the notebook
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

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: write the operation directly
a = torch.tensor([1.0, -2.0, 3.5, 0.0, 8.0], device=DEVICE, dtype=DTYPE)
b = torch.tensor([4.0, 5.0, -1.5, 7.0, -3.0], device=DEVICE, dtype=DTYPE)

c_eager = a + b
print("a:      ", a.cpu().tolist())
print("b:      ", b.cpu().tolist())
print("a + b:  ", c_eager.cpu().tolist())


# 3. PyTorch function: make it reusable
def vector_add_torch(a, b):
    return torch.add(a, b)


c_function = vector_add_torch(a, b)
torch.testing.assert_close(c_function, c_eager, rtol=0, atol=0)
print("Function:", c_function.cpu().tolist())


# 4. Write the Triton GPU kernel
@triton.jit
def vector_add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    # 1. Find the tile owned by this program.
    pid = tl.program_id(axis=0)

    # 2. Compute the logical element indices in that tile.
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 3. Load both inputs. Invalid positions receive zero locally.
    a_tile = tl.load(A + offsets, mask=valid, other=0.0)
    b_tile = tl.load(B + offsets, mask=valid, other=0.0)

    # 4. Add the tiles element by element.
    c_tile = a_tile + b_tile

    # 5. Write only real output elements.
    tl.store(C + offsets, c_tile, mask=valid)


# 5. Allocate the output and launch from Python
def vector_add_triton(a, b, block_size=256):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 1 or b.shape != a.shape:
        raise ValueError("Inputs must be vectors with the same shape.")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 inputs.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("Inputs must be contiguous.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    c = torch.empty_like(a)
    n = a.numel()
    if n == 0:
        return c

    grid = (triton.cdiv(n, block_size),)
    with torch.cuda.device(a.device):
        vector_add_kernel[grid](
            a, b, c, n, BLOCK_SIZE=block_size, num_warps=4
        )
    return c


# 6. Match output values and check the tail
c_triton = vector_add_triton(a, b)
torch.testing.assert_close(c_triton, c_eager, rtol=0, atol=0)

print("index | eager     | function  | Triton    | abs error")
for i, (e, f, t) in enumerate(zip(
    c_eager.cpu().tolist(),
    c_function.cpu().tolist(),
    c_triton.cpu().tolist(),
)):
    print(f"{i:5d} | {e:9.6f} | {f:9.6f} | {t:9.6f} | {abs(e-t):.3e}")

for n in (0, 1, 127, 128, 255, 256, 257, 1000, 4097):
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    z = torch.randn_like(x)
    expected = vector_add_torch(x, z)
    for block_size in (128, 256, 512, 1024):
        actual = vector_add_triton(x, z, block_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

print("All boundary-size and block-size checks passed.")


# 7. Compare execution times with the same inputs
def bench_ms(fn):
    # Compile and warm up before measuring steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us   function us   Triton us   speedup   max abs error")
for n in (1024, 65536, 1048576, 4194304):
    x = torch.randn(n, device=DEVICE, dtype=DTYPE)
    z = torch.randn_like(x)
    expected = vector_add_torch(x, z)
    actual = vector_add_triton(x, z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    max_error = (actual - expected).abs().max().item()

    eager_ms = bench_ms(lambda: x + z)
    function_ms = bench_ms(lambda: vector_add_torch(x, z))
    triton_ms = bench_ms(lambda: vector_add_triton(x, z))
    speedup = function_ms / triton_ms
    print(
        f"{n:8d} {eager_ms*1000:10.3f} {function_ms*1000:13.3f} "
        f"{triton_ms*1000:11.3f} {speedup:9.2f}x {max_error:14.3e}"
    )


# 8. Compare launches with preallocated output
n = 4194304
x = torch.randn(n, device=DEVICE, dtype=DTYPE)
z = torch.randn_like(x)
torch_out = torch.empty_like(x)
triton_out = torch.empty_like(x)
block_size = 256
grid = (triton.cdiv(n, block_size),)


def launch_torch_add():
    torch.add(x, z, out=torch_out)


def launch_triton_add():
    vector_add_kernel[grid](
        x, z, triton_out, n, BLOCK_SIZE=block_size, num_warps=4
    )


launch_torch_add()
launch_triton_add()
torch.testing.assert_close(triton_out, torch_out, rtol=0, atol=0)

torch_ms = bench_ms(launch_torch_add)
triton_ms = bench_ms(launch_triton_add)
logical_bytes = 3 * n * x.element_size()
print(f"PyTorch: {torch_ms*1000:.3f} us")
print(f"Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Triton effective bandwidth: {logical_bytes/(triton_ms*1e-3)/1e9:.1f} GB/s")
```
