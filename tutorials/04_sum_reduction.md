# 4. Sum reduction: combine many values into one

Previous: [fused bias + activation](03_fused_bias_activation.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run the examples in sections 1–9 in order in a fresh notebook, or run the final complete code block on its own. The final block can also be saved as a Python script. The examples use a contiguous, finite **FP32 vector on one NVIDIA CUDA GPU** and return a **zero-dimensional FP32 tensor**, matching the shape and dtype of `torch.sum(x, dtype=torch.float32)`. They implement forward computation without a registered PyTorch backward operation.

## The operation and its equation

A sum reduction combines every element of a vector into one scalar:

$$
s=\sum_{i=0}^{N-1}x_i.
$$

For an empty vector, the sum is zero. For `x = [1, 2, 3, 4, 5]`, the result is `15`.

Unlike vector addition or an activation, a reduction has a dependency across input values. Many programs cannot independently store into the same scalar without a coordination strategy.

### A local reduction tree

Pairwise addition combines values in levels. The tree below is conceptual; Triton's compiler chooses the actual mapping to GPU instructions and resources.

```text
input          1     2     3     4     5     6     7     8
                \   /      \   /      \   /      \   /
level 1           3          7         11         15
                   \       /           \        /
level 2               10                  26
                        \                /
level 3                         36
```

`tl.sum(tile, axis=0)` performs a reduction over the logical tile in **one program**. It does not synchronize all programs in a launch. See the [`tl.sum` API](https://triton-lang.org/main/python-api/generated/triton.language.sum.html).

### Reduce tiles, then reduce their partial sums

For tile size $B$, first compute

$$
p_j=\sum_{i=jB}^{\min((j+1)B,N)-1}x_i,
\qquad 0\le j<\left\lceil N/B\right\rceil.
$$

Then reduce those partial sums:

$$
s=\sum_j p_j.
$$

```text
Example: x = [1, 2, 3, 4, 5], illustrative B = 4

program 0: [1, 2, 3, 4] ── sum ── 10 ──┐
program 1: [5, 0, 0, 0] ── sum ──  5 ──┴── next kernel ── 15
               ^  ^  ^
          masked values are zero
```

If the partial-sum vector is still too large for one tile, repeat the same stage. Every program has a unique output slot. Launches use the same CUDA stream, so a later stage reads the completed earlier-stage outputs without a host synchronization between stages.

With `BLOCK_SIZE=1024`, for example:

```text
N = 1000       : 1000 ──> scalar                 (1 launch)
N = 4097       : 4097 ──> 5 ──> scalar           (2 launches)
N = 4194304    : 4194304 ──> 4096 ──> 4 ──> scalar  (3 launches)
```

A bounded tile size avoids trying to reduce an arbitrarily large input in one giant Triton tile.

## 1. Imports and input

Use a CUDA-enabled PyTorch build and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import math
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

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
```

## 2. PyTorch eager: sum the vector

```python
s_eager = torch.sum(x, dtype=torch.float32)
print("Input:", x.cpu().tolist())
print("Sum:", s_eager.item())
print("Output shape:", s_eager.shape)
```

The value is `15.0` and the shape is `torch.Size([])`. `.item()` copies the scalar to the CPU and waits for it to become available; keep it outside the timed region.

Specifying the accumulation dtype makes the reference explicit. The [PyTorch sum documentation](https://docs.pytorch.org/docs/stable/generated/torch.sum.html) explains full-tensor and dimension-specific reductions.

## 3. PyTorch function: a reusable reference

```python
def sum_torch(x):
    return torch.sum(x, dtype=torch.float32)


s_function = sum_torch(x)
torch.testing.assert_close(s_function, s_eager, rtol=0, atol=0)
print("Function result:", s_function.item())
```

This function calls the same eager reduction. PyTorch chooses its own GPU implementation; writing a Python wrapper does not change the summation algorithm.

## 4. Triton: one stage of the reduction

```python
@triton.jit
def sum_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # 1. Each program owns one input tile and one output partial sum.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 2. Zero is the identity for addition, so masked lanes contribute zero.
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)

    # 3. Reduce this program's tile to one FP32 value.
    partial = tl.sum(values, axis=0)

    # 4. Each program writes a different partial-sum slot.
    tl.store(Partials + pid, partial)
```

Step by step:

1. Assign one input tile to each program using its program ID.
2. Load valid elements and fill the remaining logical positions with zero. Unlike an elementwise kernel, masked values enter the reduction, so the fill value is essential.
3. Use `tl.sum` across the tile. The local output is a scalar rather than an elementwise tile.
4. Store the scalar at `Partials[pid]`. The grid has exactly as many programs as the output has slots.

For a one-program launch, `Partials` can point directly to the scalar output. For a larger launch, it points to a partial-sum vector. The same kernel can reduce that vector in the next stage.

There is no shared scalar updated by every program, so there is no write race and no need for `tl.atomic_add` or a zero-initialized atomic accumulator.

## 5. Allocate stages and launch the complete reduction

Separate buffer allocation from launching so the final benchmark can reuse the same scratch storage.

```python
def make_sum_buffers(x, block_size=1024):
    # Private-use helper: x and block_size satisfy sum_triton's checks.
    output = torch.empty((), device=x.device, dtype=torch.float32)
    scratch = []
    count = x.numel()
    while count > block_size:
        count = triton.cdiv(count, block_size)
        scratch.append(torch.empty(count, device=x.device, dtype=torch.float32))
    return output, scratch


def launch_sum_into(x, output, scratch, block_size=1024):
    # Private-use helper: reuse only buffers made for this input length/device.
    # Input, output, and scratch storage must be distinct.
    current = x
    count = x.numel()
    if count == 0:
        output.zero_()
        return

    with torch.cuda.device(x.device):
        for partials in scratch:
            sum_stage_kernel[(partials.numel(),)](
                current, partials, count,
                BLOCK_SIZE=block_size, num_warps=4,
            )
            current = partials
            count = partials.numel()

        # The allocation plan guarantees count <= block_size here.
        sum_stage_kernel[(1,)](
            current, output, count, BLOCK_SIZE=block_size, num_warps=4
        )


def sum_triton(x, block_size=1024):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    if x.numel() == 0:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    output, scratch = make_sum_buffers(x, block_size)
    launch_sum_into(x, output, scratch, block_size)
    return output
```

`make_sum_buffers` repeatedly applies ceiling division to determine each partial-sum vector's length. `launch_sum_into` runs one kernel for each vector and a final one-program kernel. `sum_triton` provides the validated public function and matches the empty-vector sum of zero.

All stages accumulate and store in FP32. Scratch values do not need initial zeros because each stage writes every one of its output slots before the next stage reads them. Keep the tensors alive through the launch sequence, as the `scratch` list does here.

## 6. Match values and test the reduction stages

```python
s_triton = sum_triton(x)
torch.testing.assert_close(s_triton, s_eager, rtol=0, atol=0)
print(f"Eager:    {s_eager.item():.6f}")
print(f"Function: {s_function.item():.6f}")
print(f"Triton:   {s_triton.item():.6f}")
print(f"Abs error: {(s_triton-s_eager).abs().item():.3e}")

# These signed multiples of 1/16 are exactly representable at these sizes.
# They let us check masks and stage indexing without rounding ambiguity.
for n in (0, 1, 127, 128, 255, 256, 257, 1023, 1024, 1025, 4097, 1048579):
    integers = torch.arange(n, device=DEVICE, dtype=torch.int64)
    values = (integers % 17).to(torch.float32) / 16.0 - 0.5
    expected = sum_torch(values)
    for block_size in (128, 256, 512, 1024):
        actual = sum_triton(values, block_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# A general positive FP32 input also exercises ordinary rounding behavior.
values = torch.rand(65537, device=DEVICE, dtype=DTYPE)
expected = sum_torch(values)
actual = sum_triton(values)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Random-input absolute error:", (actual - expected).abs().item())
print("All empty-input, boundary, and multistage checks passed.")
```

The hand-worked input produces `15.0` for all three paths and zero absolute error. `1048579` forces more than two stages with the default tile size. The exact-value checks also exercise negative inputs, cancellation, and different tile sizes.

## 7. Understand floating-point summation differences

Real-number addition is associative; floating-point addition can depend on the grouping:

$$
\operatorname{fl}(\operatorname{fl}(a+b)+c)
\ne \operatorname{fl}(a+\operatorname{fl}(b+c)).
$$

For example, FP32 can lose the `1` in `(100000000 + 1) - 100000000`, even though the mathematical sum is `1`. PyTorch and Triton can use different trees. A small difference between their final sums does not necessarily mean an implementation error. See [PyTorch's numerical-accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

```python
signed_values = torch.randn(65537, device=DEVICE, dtype=DTYPE)
torch_result = sum_torch(signed_values)
triton_result = sum_triton(signed_values)
reference64 = signed_values.sum(dtype=torch.float64)

print("PyTorch FP32:", torch_result.item())
print("Triton FP32: ", triton_result.item())
print("FP64 reference:", reference64.item())
print("PyTorch vs FP64 error:", (torch_result.double() - reference64).abs().item())
print("Triton vs FP64 error: ", (triton_result.double() - reference64).abs().item())

# A conservative, scale-aware diagnostic for this finite signed example.
# This is not a universal tolerance prescription for all reductions.
eps = torch.finfo(torch.float32).eps
levels = max(1, math.ceil(math.log2(signed_values.numel())))
sum_abs = signed_values.abs().sum(dtype=torch.float64).item()
diagnostic_atol = 4.0 * eps * levels * sum_abs
torch.testing.assert_close(
    triton_result, torch_result, rtol=1e-5, atol=diagnostic_atol
)
print("Diagnostic absolute tolerance:", diagnostic_atol)
```

The FP64 sum is a more accurate diagnostic reference, not exact real arithmetic. When positive and negative values nearly cancel, the result can be close to zero while the total input magnitude is large; a relative tolerance based only on the output can then be misleading. The scale-aware check above is deliberately conservative, so use it together with the strict structural checks in section 6 and the printed FP64 errors. Choose tolerances for your application's accuracy requirements.

## 8. Compare eager, function, and Triton times

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us  function us  Triton us  speedup  PyTorch sum  Triton sum  abs error")
for n in (1024, 65536, 1048576, 4194304):
    # Positive values avoid cancellation obscuring this basic comparison.
    values = torch.rand(n, device=DEVICE, dtype=DTYPE)
    expected = sum_torch(values)
    actual = sum_triton(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    ref_value = expected.item()
    got_value = actual.item()
    error = abs(got_value - ref_value)

    eager_ms = bench_ms(lambda: torch.sum(values, dtype=torch.float32))
    function_ms = bench_ms(lambda: sum_torch(values))
    triton_ms = bench_ms(lambda: sum_triton(values))
    print(
        f"{n:8d} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{ref_value:12.6f} {got_value:12.6f} {error:10.3e}"
    )
```

The Triton timing includes **all stages** of the reduction and the allocating wrapper's output and scratch-buffer behavior. Timing only the first partial-sum kernel would not measure the operation requested here. PyTorch eager and function calls also allocate their scalar output.

The table shows median GPU latency in microseconds using [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html). Inputs, first compilation, correctness assertions, CPU scalar copies, and printing are outside the timed region. These event measurements may include host dispatch gaps; they are not full CPU wall-clock measurements. No GPU timings are assumed in advance.

A speedup above `1.0x` means this Triton implementation was faster in that measurement. Multiple launches and partial-sum traffic can make this simple staged implementation slower than PyTorch's optimized reduction, particularly at small sizes.

## 9. Compare the complete reduction with reused buffers

Both paths receive preallocated scalar output storage. Triton also reuses its full scratch plan. Every invocation still runs every required stage.

```python
n = 4194304
values = torch.rand(n, device=DEVICE, dtype=DTYPE)
torch_out = torch.empty((), device=DEVICE, dtype=DTYPE)
triton_out, scratch = make_sum_buffers(values, block_size=1024)


def launch_torch_sum():
    # Specify the dimension to use the torch.sum overload accepting out=.
    torch.sum(values, dim=0, dtype=torch.float32, out=torch_out)


def launch_triton_sum():
    launch_sum_into(values, triton_out, scratch, block_size=1024)


launch_torch_sum()
launch_triton_sum()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-5, atol=1e-5)

torch_ms = bench_ms(launch_torch_sum)
triton_ms = bench_ms(launch_triton_sum)
print("Partial-buffer lengths:", [buffer.numel() for buffer in scratch])
print("Triton stages:", len(scratch) + 1)
print(f"Preallocated PyTorch sum: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton sum:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Abs error: {(triton_out-torch_out).abs().item():.3e}")
```

For this input and tile size, scratch lengths are `[4096, 4]` and there are three launches. Recreate the buffers when the input length, device, dtype, or block size changes. Output and scratch must not overlap with the input, and the buffers must not be shared by overlapping invocations on different streams.

## Try next

Experiment with tile sizes of `128`, `256`, `512`, and `1024`, rebuilding scratch buffers each time. Check both the values and the full reduction time. A further extension is a row-wise sum of a matrix, where each row produces one output rather than reducing the entire vector to a single scalar.

## 10. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–9: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports and input
import math
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

x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


# 2. PyTorch eager: sum the vector
s_eager = torch.sum(x, dtype=torch.float32)
print("Input:", x.cpu().tolist())
print("Sum:", s_eager.item())
print("Output shape:", s_eager.shape)


# 3. PyTorch function: a reusable reference
def sum_torch(x):
    return torch.sum(x, dtype=torch.float32)


s_function = sum_torch(x)
torch.testing.assert_close(s_function, s_eager, rtol=0, atol=0)
print("Function result:", s_function.item())


# 4. Triton: one stage of the reduction
@triton.jit
def sum_stage_kernel(X, Partials, N, BLOCK_SIZE: tl.constexpr):
    # 1. Each program owns one input tile and one output partial sum.
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < N

    # 2. Zero is the identity for addition, so masked lanes contribute zero.
    values = tl.load(X + offsets, mask=valid, other=0.0).to(tl.float32)

    # 3. Reduce this program's tile to one FP32 value.
    partial = tl.sum(values, axis=0)

    # 4. Each program writes a different partial-sum slot.
    tl.store(Partials + pid, partial)


# 5. Allocate stages and launch the complete reduction
def make_sum_buffers(x, block_size=1024):
    # Private-use helper: x and block_size satisfy sum_triton's checks.
    output = torch.empty((), device=x.device, dtype=torch.float32)
    scratch = []
    count = x.numel()
    while count > block_size:
        count = triton.cdiv(count, block_size)
        scratch.append(torch.empty(count, device=x.device, dtype=torch.float32))
    return output, scratch


def launch_sum_into(x, output, scratch, block_size=1024):
    # Private-use helper: reuse only buffers made for this input length/device.
    # Input, output, and scratch storage must be distinct.
    current = x
    count = x.numel()
    if count == 0:
        output.zero_()
        return

    with torch.cuda.device(x.device):
        for partials in scratch:
            sum_stage_kernel[(partials.numel(),)](
                current, partials, count,
                BLOCK_SIZE=block_size, num_warps=4,
            )
            current = partials
            count = partials.numel()

        # The allocation plan guarantees count <= block_size here.
        sum_stage_kernel[(1,)](
            current, output, count, BLOCK_SIZE=block_size, num_warps=4
        )


def sum_triton(x, block_size=1024):
    if not x.is_cuda:
        raise ValueError("Input must be a CUDA tensor.")
    if x.ndim != 1 or not x.is_contiguous():
        raise ValueError("Input must be a contiguous vector.")
    if x.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 input.")
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")

    if x.numel() == 0:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    output, scratch = make_sum_buffers(x, block_size)
    launch_sum_into(x, output, scratch, block_size)
    return output


# 6. Match values and test the reduction stages
s_triton = sum_triton(x)
torch.testing.assert_close(s_triton, s_eager, rtol=0, atol=0)
print(f"Eager:    {s_eager.item():.6f}")
print(f"Function: {s_function.item():.6f}")
print(f"Triton:   {s_triton.item():.6f}")
print(f"Abs error: {(s_triton-s_eager).abs().item():.3e}")

# These signed multiples of 1/16 are exactly representable at these sizes.
# They let us check masks and stage indexing without rounding ambiguity.
for n in (0, 1, 127, 128, 255, 256, 257, 1023, 1024, 1025, 4097, 1048579):
    integers = torch.arange(n, device=DEVICE, dtype=torch.int64)
    values = (integers % 17).to(torch.float32) / 16.0 - 0.5
    expected = sum_torch(values)
    for block_size in (128, 256, 512, 1024):
        actual = sum_triton(values, block_size)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

# A general positive FP32 input also exercises ordinary rounding behavior.
values = torch.rand(65537, device=DEVICE, dtype=DTYPE)
expected = sum_torch(values)
actual = sum_triton(values)
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
print("Random-input absolute error:", (actual - expected).abs().item())
print("All empty-input, boundary, and multistage checks passed.")


# 7. Understand floating-point summation differences
signed_values = torch.randn(65537, device=DEVICE, dtype=DTYPE)
torch_result = sum_torch(signed_values)
triton_result = sum_triton(signed_values)
reference64 = signed_values.sum(dtype=torch.float64)

print("PyTorch FP32:", torch_result.item())
print("Triton FP32: ", triton_result.item())
print("FP64 reference:", reference64.item())
print("PyTorch vs FP64 error:", (torch_result.double() - reference64).abs().item())
print("Triton vs FP64 error: ", (triton_result.double() - reference64).abs().item())

# A conservative, scale-aware diagnostic for this finite signed example.
# This is not a universal tolerance prescription for all reductions.
eps = torch.finfo(torch.float32).eps
levels = max(1, math.ceil(math.log2(signed_values.numel())))
sum_abs = signed_values.abs().sum(dtype=torch.float64).item()
diagnostic_atol = 4.0 * eps * levels * sum_abs
torch.testing.assert_close(
    triton_result, torch_result, rtol=1e-5, atol=diagnostic_atol
)
print("Diagnostic absolute tolerance:", diagnostic_atol)


# 8. Compare eager, function, and Triton times
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("N        eager us  function us  Triton us  speedup  PyTorch sum  Triton sum  abs error")
for n in (1024, 65536, 1048576, 4194304):
    # Positive values avoid cancellation obscuring this basic comparison.
    values = torch.rand(n, device=DEVICE, dtype=DTYPE)
    expected = sum_torch(values)
    actual = sum_triton(values)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    ref_value = expected.item()
    got_value = actual.item()
    error = abs(got_value - ref_value)

    eager_ms = bench_ms(lambda: torch.sum(values, dtype=torch.float32))
    function_ms = bench_ms(lambda: sum_torch(values))
    triton_ms = bench_ms(lambda: sum_triton(values))
    print(
        f"{n:8d} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{ref_value:12.6f} {got_value:12.6f} {error:10.3e}"
    )


# 9. Compare the complete reduction with reused buffers
n = 4194304
values = torch.rand(n, device=DEVICE, dtype=DTYPE)
torch_out = torch.empty((), device=DEVICE, dtype=DTYPE)
triton_out, scratch = make_sum_buffers(values, block_size=1024)


def launch_torch_sum():
    # Specify the dimension to use the torch.sum overload accepting out=.
    torch.sum(values, dim=0, dtype=torch.float32, out=torch_out)


def launch_triton_sum():
    launch_sum_into(values, triton_out, scratch, block_size=1024)


launch_torch_sum()
launch_triton_sum()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-5, atol=1e-5)

torch_ms = bench_ms(launch_torch_sum)
triton_ms = bench_ms(launch_triton_sum)
print("Partial-buffer lengths:", [buffer.numel() for buffer in scratch])
print("Triton stages:", len(scratch) + 1)
print(f"Preallocated PyTorch sum: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton sum:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Abs error: {(triton_out-torch_out).abs().item():.3e}")
```
