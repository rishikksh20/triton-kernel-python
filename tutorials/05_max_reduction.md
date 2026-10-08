# 5. Max reduction: select the largest value

Previous: [sum reduction](04_sum_reduction.md). Next: [softmax](06_softmax.md).

This is a self-contained Markdown notebook. Run the examples in sections 1–8 in order, or run the final complete code block on its own. The examples use a nonempty, contiguous **FP32 vector of finite values on one NVIDIA CUDA GPU** and return a zero-dimensional FP32 tensor. These Triton kernels implement forward computation without registering a PyTorch backward operation.

## The operation and its equation

A maximum reduction selects the largest input value:

$$
m=\max_{0\le i<N}x_i.
$$

For `x = [-2, 7, 1.5, -4, 3]`, the result is `7`. This tutorial returns the value, not its index. [`torch.max(x)`](https://docs.pytorch.org/docs/stable/generated/torch.max.html) is the matching PyTorch reference. `torch.max(x, dim=...)` additionally returns indices, which this kernel does not compute.

The reduction can combine tiles in stages:

$$
p_j=\max_{jB\le i<\min((j+1)B,N)}x_i,
\qquad m=\max_j p_j.
$$

```text
Example with illustrative tile size B = 4

[-2, 7, 1.5, -4]       -> local max 7 -> next-stage max -> 7
[ 3, -inf, -inf, -inf] -> local max 3 -------^
       ^     ^     ^
    masked positions
```

For a maximum, the padding identity is **negative infinity**, because `max(x, -inf) = x`. Padding with zero would corrupt a result such as `max([-5, -2, -9]) = -2`. This is the key change from sum reduction, where padding uses zero.

[`tl.max(tile, axis=0)`](https://triton-lang.org/main/python-api/generated/triton.language.max.html) reduces one program's tile. For a full-vector maximum, each program writes a different partial result and later launches combine those results. An empty vector has no selected maximum; this tutorial rejects it, as the full-tensor PyTorch maximum does.

## 1. Imports and example input

Use a CUDA-enabled PyTorch installation and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

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
x = torch.tensor([-2.0, 7.0, 1.5, -4.0, 3.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

## 2. PyTorch eager: compute the maximum directly

```python
m_eager = torch.max(x)
print("Input:", x.cpu().tolist())
print("Maximum:", m_eager.item())
print("Output shape:", m_eager.shape)
```

The result is `7.0`, with shape `torch.Size([])`. `.item()` waits for the scalar and copies it to the CPU, so it belongs outside the benchmark.

## 3. PyTorch function: a reusable reference

```python
def max_torch(x):
    return torch.max(x)


m_function = max_torch(x)
torch.testing.assert_close(m_function, m_eager, rtol=0, atol=0)
print("Function result:", m_function.item())
```

The function calls the same eager operation. Its Python definition does not compile a different reduction algorithm.

## 4. Triton: reduce one tile to one partial maximum

```python
@triton.jit
def max_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # Each program owns one input tile and one partial-output slot.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # Negative infinity cannot win against a valid finite input.
    values = tl.load(X + offsets, mask=valid, other=-float("inf"))
    partial = tl.max(values, axis=0)
    tl.store(Partials + pid, partial)
```

1. Build the tile indices from the program ID and a compile-time tile size.
2. Mask out-of-range loads and replace their local values with negative infinity.
3. Reduce the logical tile with `tl.max`.
4. Write the partial maximum to `Partials[pid]`, avoiding overlapping writes.

There is no global scalar concurrently updated by every program. The same kernel handles both the original vector and a vector of partial maxima.

## 5. Allocate buffers and launch all stages

```python
def make_max_buffers(x, block_size=1024):
    # Helper assumes a validated, nonempty input and supported block size.
    output = torch.empty((), device=x.device, dtype=torch.float32)
    scratch = []
    count = x.numel()
    while count > block_size:
        count = triton.cdiv(count, block_size)
        scratch.append(torch.empty(count, device=x.device, dtype=torch.float32))
    return output, scratch


def launch_max_into(x, output, scratch, block_size=1024):
    # Use distinct buffers made for this input length, device, and block size.
    current = x
    count = x.numel()
    if count == 0:
        raise ValueError("An empty vector has no maximum.")
    with torch.cuda.device(x.device):
        for partials in scratch:
            max_stage_kernel[(partials.numel(),)](
                current, partials, count, BLOCK_SIZE=block_size, num_warps=4
            )
            current = partials
            count = partials.numel()
        max_stage_kernel[(1,)](
            current, output, count, BLOCK_SIZE=block_size, num_warps=4
        )


def max_triton(x, block_size=1024):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if x.numel() == 0:
        raise ValueError("An empty vector has no maximum.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")
    output, scratch = make_max_buffers(x, block_size)
    launch_max_into(x, output, scratch, block_size)
    return output
```

The allocation plan repeatedly applies ceiling division until the remaining vector fits into one tile. For `N=4194304` and `block_size=1024`, the stages are `4194304 -> 4096 -> 4 -> scalar`: three launches, with scratch lengths `[4096, 4]`.

Each stage completely writes its output. The next launch reads those values on the same CUDA stream, so no host synchronization is needed between stages. Scratch and output do not require initialization.

## 6. Match values and check negative inputs and tail positions

```python
m_triton = max_triton(x)
torch.testing.assert_close(m_triton, m_eager, rtol=0, atol=0)
print(f"Eager: {m_eager.item():.6f}, function: {m_function.item():.6f}")
print(f"Triton: {m_triton.item():.6f}")
print("Absolute error:", (m_triton - m_eager).abs().item())

for n in (1, 127, 128, 255, 256, 257, 1023, 1024, 1025, 4097, 1048579):
    values = -torch.rand(n, device=DEVICE, dtype=DTYPE) - 0.5
    for block_size in (128, 256, 512, 1024):
        # All-negative inputs catch an incorrect zero padding value.
        torch.testing.assert_close(
            max_triton(values, block_size), max_torch(values), rtol=0, atol=0
        )
    values[-1] = 17.0  # Put the unique winner in the final position.
    torch.testing.assert_close(max_triton(values), max_torch(values), rtol=0, atol=0)

try:
    max_triton(torch.empty(0, device=DEVICE, dtype=DTYPE))
except ValueError:
    print("Empty-input rejection passed.")
else:
    raise AssertionError("The empty input should have been rejected.")
print("All value, tail, and multistage checks passed.")
```

For finite FP32 inputs, the maximum selects an existing input value rather than accumulating rounded additions. Exact value comparisons are appropriate here. NaN propagation and an argmax tie-breaking rule would need additional semantics if you extend this kernel.

## 7. Compare eager, function, and Triton times

```python
def bench_ms(fn):
    # Trigger compilation and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us  function us  Triton us  speedup  PyTorch max  Triton max  abs error")
for n in (1024, 65536, 1048576, 4194304):
    values = torch.randn(n, device=DEVICE, dtype=DTYPE)
    expected = max_torch(values)
    actual = max_triton(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ref_value, got_value = expected.item(), actual.item()
    eager_ms = bench_ms(lambda: torch.max(values))
    function_ms = bench_ms(lambda: max_torch(values))
    triton_ms = bench_ms(lambda: max_triton(values))
    print(
        f"{n:8d} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{ref_value:11.6f} {got_value:11.6f} {abs(ref_value-got_value):10.3e}"
    )
```

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; this table prints median microseconds. These are GPU event timings of the allocating API calls, which may include host dispatch gaps. Inputs, first compilation, assertions, scalar copies, and printing are outside timing. They are not full CPU wall-clock measurements.

The Triton measurement includes **every stage**, including the final scalar result. Speedup is `PyTorch function time / Triton time`; above `1.0x` means the Triton measurement is faster. Run the block for actual measurements on your GPU; the staged teaching implementation may be slower than PyTorch's optimized maximum.

## 8. Measure the complete Triton launch sequence with reused buffers

```python
values = torch.randn(4194304, device=DEVICE, dtype=DTYPE)
output, scratch = make_max_buffers(values, block_size=1024)


def launch_max_reused():
    launch_max_into(values, output, scratch, block_size=1024)


launch_max_reused()
torch.testing.assert_close(output, max_torch(values), rtol=0, atol=0)
ms = bench_ms(launch_max_reused)
print("Scratch lengths:", [buffer.numel() for buffer in scratch])
print("Stages:", len(scratch) + 1)
print(f"Triton with reused output/scratch: {ms*1000:.3f} us")
print("Maximum:", output.item())
```

This separate timing removes output and scratch allocation from the Triton callable. It is not the same API path as section 7. Rebuild the buffers when the length, device, or block size changes, and do not share them across concurrent invocations on different streams.

## Try next

Continue with [softmax](06_softmax.md), which uses a maximum reduction to stabilize exponentials and a sum reduction to normalize them.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports and example input
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
x = torch.tensor([-2.0, 7.0, 1.5, -4.0, 3.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: compute the maximum directly
m_eager = torch.max(x)
print("Input:", x.cpu().tolist())
print("Maximum:", m_eager.item())
print("Output shape:", m_eager.shape)


# 3. PyTorch function: a reusable reference
def max_torch(x):
    return torch.max(x)


m_function = max_torch(x)
torch.testing.assert_close(m_function, m_eager, rtol=0, atol=0)
print("Function result:", m_function.item())


# 4. Triton: reduce one tile to one partial maximum
@triton.jit
def max_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # Each program owns one input tile and one partial-output slot.
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # Negative infinity cannot win against a valid finite input.
    values = tl.load(X + offsets, mask=valid, other=-float("inf"))
    partial = tl.max(values, axis=0)
    tl.store(Partials + pid, partial)


# 5. Allocate buffers and launch all stages
def make_max_buffers(x, block_size=1024):
    # Helper assumes a validated, nonempty input and supported block size.
    output = torch.empty((), device=x.device, dtype=torch.float32)
    scratch = []
    count = x.numel()
    while count > block_size:
        count = triton.cdiv(count, block_size)
        scratch.append(torch.empty(count, device=x.device, dtype=torch.float32))
    return output, scratch


def launch_max_into(x, output, scratch, block_size=1024):
    # Use distinct buffers made for this input length, device, and block size.
    current = x
    count = x.numel()
    if count == 0:
        raise ValueError("An empty vector has no maximum.")
    with torch.cuda.device(x.device):
        for partials in scratch:
            max_stage_kernel[(partials.numel(),)](
                current, partials, count, BLOCK_SIZE=block_size, num_warps=4
            )
            current = partials
            count = partials.numel()
        max_stage_kernel[(1,)](
            current, output, count, BLOCK_SIZE=block_size, num_warps=4
        )


def max_triton(x, block_size=1024):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if x.numel() == 0:
        raise ValueError("An empty vector has no maximum.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")
    output, scratch = make_max_buffers(x, block_size)
    launch_max_into(x, output, scratch, block_size)
    return output


# 6. Match values and check negative inputs and tail positions
m_triton = max_triton(x)
torch.testing.assert_close(m_triton, m_eager, rtol=0, atol=0)
print(f"Eager: {m_eager.item():.6f}, function: {m_function.item():.6f}")
print(f"Triton: {m_triton.item():.6f}")
print("Absolute error:", (m_triton - m_eager).abs().item())

for n in (1, 127, 128, 255, 256, 257, 1023, 1024, 1025, 4097, 1048579):
    values = -torch.rand(n, device=DEVICE, dtype=DTYPE) - 0.5
    for block_size in (128, 256, 512, 1024):
        # All-negative inputs catch an incorrect zero padding value.
        torch.testing.assert_close(
            max_triton(values, block_size), max_torch(values), rtol=0, atol=0
        )
    values[-1] = 17.0  # Put the unique winner in the final position.
    torch.testing.assert_close(max_triton(values), max_torch(values), rtol=0, atol=0)

try:
    max_triton(torch.empty(0, device=DEVICE, dtype=DTYPE))
except ValueError:
    print("Empty-input rejection passed.")
else:
    raise AssertionError("The empty input should have been rejected.")
print("All value, tail, and multistage checks passed.")


# 7. Compare eager, function, and Triton times
def bench_ms(fn):
    # Trigger compilation and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us  function us  Triton us  speedup  PyTorch max  Triton max  abs error")
for n in (1024, 65536, 1048576, 4194304):
    values = torch.randn(n, device=DEVICE, dtype=DTYPE)
    expected = max_torch(values)
    actual = max_triton(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    ref_value, got_value = expected.item(), actual.item()
    eager_ms = bench_ms(lambda: torch.max(values))
    function_ms = bench_ms(lambda: max_torch(values))
    triton_ms = bench_ms(lambda: max_triton(values))
    print(
        f"{n:8d} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{ref_value:11.6f} {got_value:11.6f} {abs(ref_value-got_value):10.3e}"
    )


# 8. Measure the complete Triton launch sequence with reused buffers
values = torch.randn(4194304, device=DEVICE, dtype=DTYPE)
output, scratch = make_max_buffers(values, block_size=1024)


def launch_max_reused():
    launch_max_into(values, output, scratch, block_size=1024)


launch_max_reused()
torch.testing.assert_close(output, max_torch(values), rtol=0, atol=0)
ms = bench_ms(launch_max_reused)
print("Scratch lengths:", [buffer.numel() for buffer in scratch])
print("Stages:", len(scratch) + 1)
print(f"Triton with reused output/scratch: {ms*1000:.3f} us")
print("Maximum:", output.item())
```
