# 9. MatMul: multiply matrices with two-dimensional tiles

Previous: [LayerNorm](08_layernorm.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. The examples multiply two contiguous **FP32 matrices on one NVIDIA CUDA GPU**, with finite values whose products and accumulated sums remain representable in FP32. The output is FP32. These kernels implement forward computation without a registered PyTorch backward operation.

## The operation and its equation

Matrix multiplication combines rows of the first matrix with columns of the second. For $A\in\mathbb{R}^{M\times K}$ and $B\in\mathbb{R}^{K\times N}$:

$$
C=AB\in\mathbb{R}^{M\times N},\qquad
C_{m,n}=\sum_{k=0}^{K-1}A_{m,k}B_{k,n}.
$$

The shared dimension $K$ must match. Matrix multiplication is not elementwise multiplication: `A @ B` and `A * B` describe different operations.

```text
A: (2, 3)             B: (3, 2)          C: (2, 2)
[ [1, 2, 3],          [ [ 7,  8],        [ [ 58,  64],
  [4, 5, 6] ]           [ 9, 10],          [139, 154] ]
                        [11, 12] ]

C[0,0] = 1*7 + 2*9 + 3*11 = 58
C[1,1] = 4*8 + 5*10 + 6*12 = 154
```

Each output is a dot product. Instead of loading its inputs separately for every scalar result, a tiled kernel computes many dot products together and reuses loaded values within the tile.

## Tile the output and loop over K

A program owns a `BLOCK_M × BLOCK_N` output tile. On each K iteration, it loads a `BLOCK_M × BLOCK_K` tile of A and a `BLOCK_K × BLOCK_N` tile of B:

$$
C_{\mathrm{tile}}\leftarrow C_{\mathrm{tile}}
+A_{\mathrm{tile}}B_{\mathrm{tile}}.
$$

```text
                     B tile: BLOCK_K × BLOCK_N
                              │
A tile: BLOCK_M × BLOCK_K ── multiply ──> accumulator: BLOCK_M × BLOCK_N
                              │
                    repeat for the next K tile
                              │
                    store the completed C tile
```

One program keeps ownership of its output tile through the entire K loop. No output atomics are needed. Separate masks protect rows, columns, and the final partial K tile.

This guide uses FP32 inputs with Triton's `input_precision="ieee"` and PyTorch's highest FP32 matmul precision. That makes the numerical comparison explicit. Faster Tensor Core modes, FP16/BF16 inputs, and autotuning are later extensions. See the [`tl.dot` precision options](https://triton-lang.org/main/python-api/generated/triton.language.dot.html) and the [official tiled-matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).

## 1. Imports, precision, and example matrices

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

# Match the kernel's FP32 IEEE path rather than a reduced-precision TF32 path.
torch.set_float32_matmul_precision("highest")
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=DEVICE, dtype=DTYPE)
b = torch.tensor([[7., 8.], [9., 10.], [11., 12.]], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
print("FP32 matmul precision:", torch.get_float32_matmul_precision())
```

The precision setting affects PyTorch's FP32 matmuls in this notebook. See [`torch.set_float32_matmul_precision`](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html). Inputs remain FP32 throughout this example.

## 2. PyTorch eager: multiply the matrices

```python
c_eager = a @ b
print("A:\n", a.cpu())
print("B:\n", b.cpu())
print("A @ B:\n", c_eager.cpu())

# Expose one dot product from the equation.
c00 = (a[0, :] * b[:, 0]).sum()
torch.testing.assert_close(c00, c_eager[0, 0], rtol=0, atol=0)
print("C[0,0] from a dot product:", c00.item())
```

`a @ b` invokes the optimized PyTorch matrix multiplication. The separate scalar dot product explains one output entry; it is not the performance baseline. The expected full output is `[[58, 64], [139, 154]]`.

## 3. PyTorch function: a reusable matrix multiplication

```python
def matmul_torch(a, b):
    return torch.matmul(a, b)


c_function = matmul_torch(a, b)
torch.testing.assert_close(c_function, c_eager, rtol=0, atol=0)
print("Function output:\n", c_function.cpu())
```

For these two-dimensional inputs, the function performs the same operation as `a @ b`. The tutorial's custom wrapper supports two matrices, not the batching and broadcasting options of general `torch.matmul`.

## 4. Triton: accumulate an output tile across K

```python
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K,
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

    c_ptrs = C + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)
```

Step by step:

1. Get two program IDs: one selects a group of output rows, the other a group of output columns.
2. Use `[:, None]` and `[None, :]` to broadcast row and column offsets into two-dimensional pointer tiles.
3. Initialize an FP32 accumulator with zeros.
4. Load the next pair of A/B tiles, padding invalid values with zero so they contribute no products.
5. Use `tl.dot` to add their matrix product into the accumulator.
6. After the entire K dimension has been covered, store the valid output region once.

The loop implements the sum over K; it is not one separate GPU launch per K tile. A value loaded from A can contribute across several output columns, and a value from B across several output rows.

## 5. Python wrapper: validate dimensions and create the grid

```python
def matmul_triton(a, b):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected matrices with matching shared K dimensions.")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 inputs.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("Inputs must be contiguous.")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return c
    if k == 0:
        # Empty dot products sum to zero; every output must be initialized.
        return c.zero_()
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(a.device):
        matmul_kernel[grid](
            a, b, c, m, n, k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4,
        )
    return c
```

The grid covers the output matrix, while the kernel loop covers K. Empty output dimensions return an empty tensor. A positive-sized output with `K=0` is explicitly filled with zero, matching the sum over an empty shared dimension.

The fixed `32 × 32 × 32` tile configuration keeps the example understandable. Tile sizes and `num_warps` are tuning choices, not matrix dimensions or individual CUDA thread counts.

## 6. Compare output entries and test all three boundaries

```python
c_triton = matmul_triton(a, b)
torch.testing.assert_close(c_triton, c_eager, rtol=0, atol=0)
print("row, column | PyTorch | Triton | abs error")
for r, (references, results) in enumerate(zip(c_eager.cpu().tolist(), c_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:.6f} | {got:.6f} | {abs(ref-got):.3e}")

for m, n, k in ((0, 7, 3), (3, 0, 7), (3, 5, 0), (1, 1, 1), (17, 19, 23), (33, 65, 31), (64, 33, 65)):
    left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
    actual = matmul_triton(left, right)
    expected = matmul_torch(left, right)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
print("All M/N/K tail and empty-dimension checks passed.")
```

The small integer example is exactly representable and checks exact equality. Random FP32 dot products need tolerances because accumulation order can differ from the library implementation, even with matching precision modes. Odd values of M, N, and K independently exercise all three masks.

For a strict application requirement, evaluate error against a higher-precision reference and choose the tolerances accordingly. The example tolerances accommodate the demonstrated moderate random inputs; they are not a universal numerical guarantee.

## 7. Compare eager, function, and Triton times and throughput

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("M,N,K           eager us  function us  Triton us  speedup  Triton TFLOP/s  max error")
for m, n, k in ((64, 64, 64), (128, 256, 64), (256, 256, 256), (512, 512, 512)):
    left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
    expected, actual = matmul_torch(left, right), matmul_triton(left, right)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: left @ right)
    function_ms = bench_ms(lambda: matmul_torch(left, right))
    triton_ms = bench_ms(lambda: matmul_triton(left, right))
    tflops = 2.0 * m * n * k / (triton_ms * 1e-3) / 1e12
    shape = f"{m},{n},{k}"
    print(
        f"{shape:15s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{tflops:14.4f} {error:10.3e}"
    )
```

The conventional matmul work estimate is $2MNK$ FLOPs, counting a multiply and an add separately. Throughput is that estimate divided by the measured time; it is not a hardware utilization counter.

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds, converted here to median microseconds. All three calls allocate their outputs. The GPU event timing may include host dispatch gaps; inputs, first compilation, correctness checks, CPU copies, and printing are outside timing. Full CPU wall-clock cost is a separate measurement. These blocks generate actual timing values when run on your GPU.

PyTorch uses a highly optimized matrix-multiplication library. This basic FP32 IEEE kernel is a learning implementation, so a speedup is not assumed. Changing Triton to TF32 or changing inputs to FP16/BF16 changes the accuracy/performance comparison and requires matching PyTorch settings and fresh correctness checks.

## 8. Compare preallocated output launches

```python
m, n, k = 512, 512, 512
left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
torch_out = torch.empty(m, n, device=DEVICE, dtype=DTYPE)
triton_out = torch.empty_like(torch_out)
grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))


def launch_torch_matmul():
    torch.mm(left, right, out=torch_out)


def launch_triton_matmul():
    matmul_kernel[grid](
        left, right, triton_out, m, n, k,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4,
    )


launch_torch_matmul()
launch_triton_matmul()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-4, atol=1e-3)
torch_ms, triton_ms = bench_ms(launch_torch_matmul), bench_ms(launch_triton_matmul)
print(f"Preallocated PyTorch: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Max absolute error: {(triton_out-torch_out).abs().max().item():.3e}")
```

This times launch paths with output storage allocated in advance. Inputs have `requires_grad=False`. It uses the same precision modes and complete K loop as the allocating comparison.

## Try next

Experiment with odd matrix dimensions, then study the [official matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) for grouped program ordering, multiple tile configurations, and autotuning. Keep dtype, precision, numerical checks, and the benchmark method explicit when comparing configurations.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all the code from sections 1–8: imports, PyTorch references, Triton kernels and launch helpers, value checks, and timing comparisons. Comments mark each step.

```python
# 1. Imports, precision, and example matrices
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

# Match the kernel's FP32 IEEE path rather than a reduced-precision TF32 path.
torch.set_float32_matmul_precision("highest")
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=DEVICE, dtype=DTYPE)
b = torch.tensor([[7., 8.], [9., 10.], [11., 12.]], device=DEVICE, dtype=DTYPE)
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
print("FP32 matmul precision:", torch.get_float32_matmul_precision())


# 2. PyTorch eager: multiply the matrices
c_eager = a @ b
print("A:\n", a.cpu())
print("B:\n", b.cpu())
print("A @ B:\n", c_eager.cpu())

# Expose one dot product from the equation.
c00 = (a[0, :] * b[:, 0]).sum()
torch.testing.assert_close(c00, c_eager[0, 0], rtol=0, atol=0)
print("C[0,0] from a dot product:", c00.item())


# 3. PyTorch function: a reusable matrix multiplication
def matmul_torch(a, b):
    return torch.matmul(a, b)


c_function = matmul_torch(a, b)
torch.testing.assert_close(c_function, c_eager, rtol=0, atol=0)
print("Function output:\n", c_function.cpu())


# 4. Triton: accumulate an output tile across K
@triton.jit
def matmul_kernel(
    A, B, C, M, N, K,
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

    c_ptrs = C + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)


# 5. Python wrapper: validate dimensions and create the grid
def matmul_triton(a, b):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected matrices with matching shared K dimensions.")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("This tutorial supports FP32 inputs.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("Inputs must be contiguous.")
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return c
    if k == 0:
        # Empty dot products sum to zero; every output must be initialized.
        return c.zero_()
    block_m, block_n, block_k = 32, 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(a.device):
        matmul_kernel[grid](
            a, b, c, m, n, k,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=4,
        )
    return c


# 6. Compare output entries and test all three boundaries
c_triton = matmul_triton(a, b)
torch.testing.assert_close(c_triton, c_eager, rtol=0, atol=0)
print("row, column | PyTorch | Triton | abs error")
for r, (references, results) in enumerate(zip(c_eager.cpu().tolist(), c_triton.cpu().tolist())):
    for c, (ref, got) in enumerate(zip(references, results)):
        print(f"({r}, {c}) | {ref:.6f} | {got:.6f} | {abs(ref-got):.3e}")

for m, n, k in ((0, 7, 3), (3, 0, 7), (3, 5, 0), (1, 1, 1), (17, 19, 23), (33, 65, 31), (64, 33, 65)):
    left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
    actual = matmul_triton(left, right)
    expected = matmul_torch(left, right)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
print("All M/N/K tail and empty-dimension checks passed.")


# 7. Compare eager, function, and Triton times and throughput
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(
        fn, warmup=25, rep=100, return_mode="median"
    )


print("M,N,K           eager us  function us  Triton us  speedup  Triton TFLOP/s  max error")
for m, n, k in ((64, 64, 64), (128, 256, 64), (256, 256, 256), (512, 512, 512)):
    left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
    right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
    expected, actual = matmul_torch(left, right), matmul_triton(left, right)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    error = (actual - expected).abs().max().item()
    eager_ms = bench_ms(lambda: left @ right)
    function_ms = bench_ms(lambda: matmul_torch(left, right))
    triton_ms = bench_ms(lambda: matmul_triton(left, right))
    tflops = 2.0 * m * n * k / (triton_ms * 1e-3) / 1e12
    shape = f"{m},{n},{k}"
    print(
        f"{shape:15s} {eager_ms*1000:9.3f} {function_ms*1000:12.3f} "
        f"{triton_ms*1000:10.3f} {function_ms/triton_ms:7.2f}x "
        f"{tflops:14.4f} {error:10.3e}"
    )


# 8. Compare preallocated output launches
m, n, k = 512, 512, 512
left = torch.randn(m, k, device=DEVICE, dtype=DTYPE)
right = torch.randn(k, n, device=DEVICE, dtype=DTYPE)
torch_out = torch.empty(m, n, device=DEVICE, dtype=DTYPE)
triton_out = torch.empty_like(torch_out)
grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))


def launch_torch_matmul():
    torch.mm(left, right, out=torch_out)


def launch_triton_matmul():
    matmul_kernel[grid](
        left, right, triton_out, m, n, k,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4,
    )


launch_torch_matmul()
launch_triton_matmul()
torch.testing.assert_close(triton_out, torch_out, rtol=1e-4, atol=1e-3)
torch_ms, triton_ms = bench_ms(launch_torch_matmul), bench_ms(launch_triton_matmul)
print(f"Preallocated PyTorch: {torch_ms*1000:.3f} us")
print(f"Preallocated Triton:  {triton_ms*1000:.3f} us")
print(f"Speedup: {torch_ms/triton_ms:.2f}x")
print(f"Max absolute error: {(triton_out-torch_out).abs().max().item():.3e}")
```
