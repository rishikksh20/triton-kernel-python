# 11. Batched MatMul: independent matrix pairs

Previous: [MatMul + bias + activation](10_matmul_bias_activation.md). Next: [attention without Flash](12_attention_without_flash.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous, finite FP32 CUDA tensors on one NVIDIA GPU; products and sums must remain representable in FP32. The examples implement forward computation without a registered backward operation.

## Concepts, shapes, and equations

Batched matrix multiplication applies the same operation to several independent matrix pairs. For batch size $B$, inputs have shapes $(B,M,K)$ and $(B,K,N)$, and the output has shape $(B,M,N)$:

$$
C_{b,m,n}=\sum_{k=0}^{K-1}A_{b,m,k}B_{b,k,n}.
$$

There is no sum over the batch dimension. Output batch `b` uses only input batch `b`. The same tiled matmul algorithm can be repeated independently across batches.

```text
A[0] @ B[0] -> C[0]
A[1] @ B[1] -> C[1]
A[2] @ B[2] -> C[2]
```

[`torch.bmm`](https://docs.pytorch.org/docs/stable/generated/torch.bmm.html) takes exactly two rank-three tensors with the same batch size; it does not broadcast batches. General `torch.matmul` has broader broadcasting semantics, which this guide does not implement.

### Batch addressing in contiguous storage

A matrix pair occupies `M*K` and `K*N` consecutive values. The batch offset is therefore `b*M*K` for A, `b*K*N` for B, and `b*M*N` for C. Within that matrix, the row/column address formulas are unchanged.

```text
A storage: [batch 0: M*K values][batch 1: M*K values]...
                      ^
base for batch b = A + b*M*K
```

The launch grid has three axes: output row tiles, output column tiles, and batches. Every program owns one output tile in one batch. Programs never mix input values or write output slots across batches.

This example limits batch count to 65535 because it maps batches to CUDA's grid-z dimension. A flattened grid can remove that particular mapping constraint. It retains the explicit FP32 IEEE precision setting from the standalone MatMul guide.

## 1. Imports and example batches

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
a = torch.tensor([[[1.,2.],[3.,4.]], [[2.,0.],[0.,2.]]],device=DEVICE,dtype=DTYPE)
b = torch.tensor([[[5.,6.],[7.,8.]], [[1.,3.],[4.,2.]]],device=DEVICE,dtype=DTYPE)
```

Both inputs have shape `(2,2,2)`: two separate matrix pairs.

## 2. PyTorch eager: multiply each pair

Use `torch.bmm` to express the full batch.

```python
c_eager = torch.bmm(a,b)
print("Batched output:\n", c_eager.cpu())
for batch in range(a.shape[0]):
    torch.testing.assert_close(c_eager[batch],a[batch] @ b[batch],rtol=0,atol=0)
```

The output batches are `[[19,22],[43,50]]` and `[[2,6],[8,4]]`. Checking each pair independently verifies the meaning of the batch dimension.

## 3. PyTorch function: a reusable batched operation

A Python function still calls the same eager implementation.

```python
def batched_matmul_torch(a,b):
    return torch.bmm(a,b)


torch.testing.assert_close(batched_matmul_torch(a,b),c_eager,rtol=0,atol=0)
```

Unlike a Python loop over `a[batch] @ b[batch]`, the built-in submits a batched operation.

## 4. Triton kernel: add a batch program ID

The K loop and edge masks are unchanged from a single matrix multiplication.

```python
@triton.jit
def batched_matmul_kernel(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + batch * M * K + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + batch * K * N + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    c_ptrs = C + batch * M * N + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)
```

1. Read `tl.program_id(2)` to select a batch.
2. Add the batch offsets to all A, B, and C pointer tiles.
3. Accumulate this batch's output tile across K.
4. Store into this batch's own output region.

The batch dimension is an independent parallel-work dimension, not a reduction dimension.

## 5. Python wrapper: launch a three-dimensional grid

The wrapper implements strict paired batches.

```python
def batched_matmul_triton(a,b):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError("Expected equal batches and matching K dimensions.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (a,b)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    batches,m,k = a.shape
    n = b.shape[2]
    if batches > 65535:
        raise ValueError("This grid-z mapping supports at most 65535 batches.")
    c = torch.empty((batches,m,n),device=a.device,dtype=torch.float32)
    if batches == 0 or m == 0 or n == 0:
        return c
    if k == 0:
        return c.zero_()
    with torch.cuda.device(a.device):
        batched_matmul_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),batches)](
            a,b,c,m,n,k,BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return c
```

Empty batches or output dimensions skip the launch. A zero K dimension produces zero-valued dot products.

## 6. Compare values, batch offsets, and tile tails

Use different contents in each batch and irregular matrix dimensions.

```python
c_triton = batched_matmul_triton(a,b)
torch.testing.assert_close(c_triton,c_eager,rtol=0,atol=0)
print("PyTorch:",c_eager.cpu().tolist())
print("Triton: ",c_triton.cpu().tolist())
print("Max absolute error:",(c_triton-c_eager).abs().max().item())
for batches,m,n,k in ((0,3,5,7),(2,0,5,7),(2,3,0,7),(2,3,5,0),(1,1,1,1),(3,17,19,23),(2,33,65,31)):
    left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
    torch.testing.assert_close(
        batched_matmul_triton(left,right),batched_matmul_torch(left,right),rtol=1e-4,atol=1e-3,
    )
print("All batch-offset, M/N/K-tail, and empty-shape checks passed.")
```

A missing batch offset can accidentally reproduce batch zero in every output. Distinct batch data catches this error. Random FP32 accumulation needs tolerances.

## 7. Compare eager, function, and Triton times

Measure the complete batch, not a single matrix pair.

```python
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("B,M,N,K        eager us  function us  Triton us  speedup  max error")
for batches,m,n,k in ((1,64,64,64),(4,64,128,64),(8,128,128,128)):
    left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
    expected,actual = batched_matmul_torch(left,right),batched_matmul_triton(left,right)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.bmm(left,right))
    p = bench_ms(lambda: batched_matmul_torch(left,right))
    t = bench_ms(lambda: batched_matmul_triton(left,right))
    print(f"{batches},{m},{n},{k} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

Each timed call allocates its output. Input creation, first compilation, assertions, CPU copies, and printing are outside timing. [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock time. A speedup above `1.0x` means the Triton measurement was faster. Run the code for actual timings; no speedup is assumed. The conventional work estimate is $2BMNK$ FLOPs.

## 8. Compare preallocated batch outputs

Both output buffers are created before timing.

```python
batches,m,n,k = 4,128,128,128
left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
torch_out = torch.empty(batches,m,n,device=DEVICE,dtype=DTYPE)
triton_out = torch.empty_like(torch_out)


def launch_torch_bmm():
    torch.bmm(left,right,out=torch_out)


def launch_triton_bmm():
    batched_matmul_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),batches)](
        left,right,triton_out,m,n,k,BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
    )


launch_torch_bmm()
launch_triton_bmm()
torch.testing.assert_close(triton_out,torch_out,rtol=1e-4,atol=1e-3)
p,t = bench_ms(launch_torch_bmm),bench_ms(launch_triton_bmm)
print(f"Preallocated PyTorch: {p*1000:.3f} us; Triton: {t*1000:.3f} us; speedup: {p/t:.2f}x")
```

Inputs have no gradients enabled. This launch comparison excludes output allocation and is distinct from the allocating API comparison above.

## Try next

Continue with [attention without Flash](12_attention_without_flash.md), which uses batched matrix multiplication twice, with row-wise softmax between them.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code: imports, references, kernels, launch helpers, correctness checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and example batches
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
a = torch.tensor([[[1.,2.],[3.,4.]], [[2.,0.],[0.,2.]]],device=DEVICE,dtype=DTYPE)
b = torch.tensor([[[5.,6.],[7.,8.]], [[1.,3.],[4.,2.]]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: multiply each pair
c_eager = torch.bmm(a,b)
print("Batched output:\n", c_eager.cpu())
for batch in range(a.shape[0]):
    torch.testing.assert_close(c_eager[batch],a[batch] @ b[batch],rtol=0,atol=0)


# 3. PyTorch function: a reusable batched operation
def batched_matmul_torch(a,b):
    return torch.bmm(a,b)


torch.testing.assert_close(batched_matmul_torch(a,b),c_eager,rtol=0,atol=0)


# 4. Triton kernel: add a batch program ID
@triton.jit
def batched_matmul_kernel(
    A, B, C, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + batch * M * K + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + batch * K * N + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    c_ptrs = C + batch * M * N + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)


# 5. Python wrapper: launch a three-dimensional grid
def batched_matmul_triton(a,b):
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        raise ValueError("Expected equal batches and matching K dimensions.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (a,b)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    batches,m,k = a.shape
    n = b.shape[2]
    if batches > 65535:
        raise ValueError("This grid-z mapping supports at most 65535 batches.")
    c = torch.empty((batches,m,n),device=a.device,dtype=torch.float32)
    if batches == 0 or m == 0 or n == 0:
        return c
    if k == 0:
        return c.zero_()
    with torch.cuda.device(a.device):
        batched_matmul_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),batches)](
            a,b,c,m,n,k,BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return c


# 6. Compare values, batch offsets, and tile tails
c_triton = batched_matmul_triton(a,b)
torch.testing.assert_close(c_triton,c_eager,rtol=0,atol=0)
print("PyTorch:",c_eager.cpu().tolist())
print("Triton: ",c_triton.cpu().tolist())
print("Max absolute error:",(c_triton-c_eager).abs().max().item())
for batches,m,n,k in ((0,3,5,7),(2,0,5,7),(2,3,0,7),(2,3,5,0),(1,1,1,1),(3,17,19,23),(2,33,65,31)):
    left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
    torch.testing.assert_close(
        batched_matmul_triton(left,right),batched_matmul_torch(left,right),rtol=1e-4,atol=1e-3,
    )
print("All batch-offset, M/N/K-tail, and empty-shape checks passed.")


# 7. Compare eager, function, and Triton times
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("B,M,N,K        eager us  function us  Triton us  speedup  max error")
for batches,m,n,k in ((1,64,64,64),(4,64,128,64),(8,128,128,128)):
    left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
    right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
    expected,actual = batched_matmul_torch(left,right),batched_matmul_triton(left,right)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.bmm(left,right))
    p = bench_ms(lambda: batched_matmul_torch(left,right))
    t = bench_ms(lambda: batched_matmul_triton(left,right))
    print(f"{batches},{m},{n},{k} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Compare preallocated batch outputs
batches,m,n,k = 4,128,128,128
left = torch.randn(batches,m,k,device=DEVICE,dtype=DTYPE)
right = torch.randn(batches,k,n,device=DEVICE,dtype=DTYPE)
torch_out = torch.empty(batches,m,n,device=DEVICE,dtype=DTYPE)
triton_out = torch.empty_like(torch_out)


def launch_torch_bmm():
    torch.bmm(left,right,out=torch_out)


def launch_triton_bmm():
    batched_matmul_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),batches)](
        left,right,triton_out,m,n,k,BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
    )


launch_torch_bmm()
launch_triton_bmm()
torch.testing.assert_close(triton_out,torch_out,rtol=1e-4,atol=1e-3)
p,t = bench_ms(launch_torch_bmm),bench_ms(launch_triton_bmm)
print(f"Preallocated PyTorch: {p*1000:.3f} us; Triton: {t*1000:.3f} us; speedup: {p/t:.2f}x")
```
