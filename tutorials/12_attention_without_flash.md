# 12. Attention without Flash: scores, softmax, and weighted values

Previous: [batched MatMul](11_batched_matmul.md). Next: [online softmax](13_online_softmax.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous, finite FP32 CUDA tensors on one NVIDIA GPU; products and sums must remain representable in FP32. The examples implement forward computation without a registered backward operation. This guide uses no dropout and implements optional causal masking only for square self-attention.

## Concepts: what queries, keys, and values do

Attention lets each query position build an output from several value vectors. A **query** describes what the position seeks, a **key** describes what a source position offers, and a **value** is the information to blend. A dot product between a query and a key is a compatibility score. Softmax converts the row of scores into weights; the final output is the weighted sum of values.

For each batch and head, use these shapes:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| Q | $(M,D)$ | M queries with D features |
| K | $(N,D)$ | N keys with the same feature dimension |
| V | $(N,D_v)$ | One value vector per key |
| Scores and probabilities | $(M,N)$ | One weight candidate per query/key pair |
| Output | $(M,D_v)$ | One blended value vector per query |

The full tensors have leading batch/head dimensions `(B,H,...)`. These dimensions describe independent attention problems. `M=N` for ordinary self-attention; the unmasked code also supports cross-attention with different query and key lengths.

### The three equations

$$
S=\frac{QK^T}{\sqrt D},\qquad
P_{i,j}=\frac{e^{S_{i,j}-m_i}}{\sum_t e^{S_{i,t}-m_i}},\quad m_i=\max_t S_{i,t},\qquad
O=PV.
$$

The output entry is $O_{i,d}=\sum_j P_{i,j}V_{j,d}$. Softmax is taken over **keys**, so each query's probability row sums to one. It is not taken over the query dimension or the value-feature dimension.

Scaling by $1/\sqrt D$ controls the typical score magnitude as the dot-product dimension grows. For independent zero-mean components with unit variance, the sum of D products has variance D; dividing by $\sqrt D$ gives variance one. This helps prevent increasingly sharp softmax distributions solely from increasing the feature count.

```text
Q (M,D) × Kᵀ (D,N) -> Scores (M,N)
                               |
                       scale and optional mask
                               |
                       softmax across N keys
                               |
                       P (M,N) × V (N,Dv) -> O (M,Dv)
```

### Causal masking

In causal self-attention, query position `i` can use keys `j <= i`. Replace future-key scores with negative infinity **before** softmax so their exponential weights are zero. Masking probabilities after normalization without renormalizing would lose the row-sum-one property.

```text
query/key       0     1     2
0             allow mask  mask
1             allow allow mask
2             allow allow allow
```

The tutorial requires `M=N` for this causal convention. Each query can at least attend to itself, so no probability row is fully masked. Arbitrary attention masks, dropout, grouped-query attention, caching, and backward gradients are separate extensions.

### Why materialize scores and probabilities?

This implementation makes the three mathematical stages visible: one Triton kernel writes scores, a second writes probabilities, and a third multiplies those probabilities by V. Both intermediate matrices live in GPU memory. For FP32, keeping both consumes $2BHMN\times4$ bytes, excluding Q/K/V/output and any additional memory.

For square sequence length T, this storage grows as $T^2$. The leading arithmetic grows as $2BHT^2(D+D_v)$ FLOPs. Merely tiling each matmul does not remove the full score/probability matrices between stages.

The PyTorch reference below uses explicit matmul and softmax calls. It does not use `scaled_dot_product_attention`, which can select different fused backends. This keeps the baseline's intermediates visible; see [PyTorch's SDPA API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) for that broader API. The next guide explains the streaming normalizer that helps support memory-efficient attention implementations.

## 1. Imports and tiny query/key/value tensors

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
q = torch.tensor([[[[1.,0.],[0.,1.]]]],device=DEVICE,dtype=DTYPE)
k = q.clone()
v = torch.tensor([[[[10.,0.],[0.,20.]]]],device=DEVICE,dtype=DTYPE)
```

Here `B=H=1`, `M=N=D=Dv=2`.

## 2. PyTorch eager: expose every attention stage

Inspect the scores and weights, not only the final output.

```python
scores = (q @ k.transpose(-1,-2)) / (q.shape[-1] ** 0.5)
probabilities = torch.softmax(scores,dim=-1)
o_eager = probabilities @ v
print("Scores:\n",scores.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())
```

The diagonal score is `1/sqrt(2)` and the off-diagonal score is zero. Each probability row favors the value corresponding to the matching key.

## 3. PyTorch function: explicit attention with optional causality

The function preserves the three eager stages.

```python
def attention_torch(q,k,v,causal=False,return_weights=False):
    scores = (q @ k.transpose(-1,-2)) / (q.shape[-1] ** 0.5)
    if causal:
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("This example uses square causal self-attention.")
        keep = torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep,-float("inf"))
    probabilities = torch.softmax(scores,dim=-1)
    output = probabilities @ v
    return (output,probabilities) if return_weights else output


torch.testing.assert_close(attention_torch(q,k,v),o_eager,rtol=0,atol=0)
print("Causal output:\n",attention_torch(q,k,v,causal=True).cpu())
```

For the first causal query, only key zero is allowed, so its output is exactly the first value `[10,0]`.

## 4. Triton: three separate kernels

The value kernel is a batched matmul; the score kernel presents K transposed through its pointer offsets.

```python
@triton.jit
def attention_scores_kernel(
    Q, Key, Scores, M, N, D, Scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten (batch, head) into one independent matrix pair.
    bh = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0,BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0,BLOCK_N)
    local_d = tl.arange(0,BLOCK_K)
    acc = tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    for d_tile in range(tl.cdiv(D,BLOCK_K)):
        d = d_tile * BLOCK_K + local_d
        q_ptrs = Q + bh * M * D + rows[:,None] * D + d[None,:]
        # Key is stored as (N,D); this pointer tile presents its transpose.
        k_ptrs = Key + bh * N * D + columns[None,:] * D + d[:,None]
        q = tl.load(q_ptrs,mask=(rows[:,None]<M)&(d[None,:]<D),other=0.0)
        k = tl.load(k_ptrs,mask=(columns[None,:]<N)&(d[:,None]<D),other=0.0)
        acc = tl.dot(q,k,acc,input_precision="ieee")
    scores = acc * Scale
    if CAUSAL:
        scores = tl.where(columns[None,:] <= rows[:,None],scores,-float("inf"))
    tl.store(Scores + bh*M*N + rows[:,None]*N + columns[None,:],scores,
             mask=(rows[:,None]<M)&(columns[None,:]<N))


@triton.jit
def attention_softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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



@triton.jit
def attention_value_kernel(
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

1. Compute scaled query/key scores, masking future positions when requested.
2. Reduce and normalize each complete score row.
3. Multiply the stored probability matrix by the value matrix.

The softmax kernel permits the internally generated masked `-inf` entries because every row still has at least one finite score. It pads extra columns with `-inf` as well. These are three launches, not a fused FlashAttention kernel.

## 5. Python wrapper: validate head shapes and allocate intermediates

The wrapper explicitly creates both `(B,H,M,N)` intermediate matrices.

```python
def attention_triton(q,k,v,causal=False,return_weights=False):
    if not all(t.is_cuda and t.device == q.device for t in (q,k,v)):
        raise ValueError("Inputs must be CUDA tensors on one device.")
    if not all(t.ndim == 4 for t in (q,k,v)):
        raise ValueError("Expected Q/K/V shaped (batch,head,length,features).")
    batches,heads,m,d = q.shape
    n = k.shape[-2]
    dv = v.shape[-1]
    if k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2] or k.shape[-1] != d or v.shape[-2] != n:
        raise ValueError("Batch/head, key/value length, and Q/K feature dimensions must match.")
    if d == 0 or dv == 0 or not 1 <= n <= 4096:
        raise ValueError("Use positive feature dimensions and 1–4096 keys.")
    if causal and m != n:
        raise ValueError("This example uses square causal self-attention.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    bh = batches * heads
    if bh > 65535:
        raise ValueError("This grid-z mapping supports at most 65535 batch/head pairs.")
    scores = torch.empty((batches,heads,m,n),device=q.device,dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    output = torch.empty((batches,heads,m,dv),device=q.device,dtype=torch.float32)
    if bh == 0 or m == 0:
        return (output,probabilities) if return_weights else output
    width = triton.next_power_of_2(n)
    with torch.cuda.device(q.device):
        attention_scores_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),bh)](
            q,k,scores,m,n,d,d**-0.5,CAUSAL=causal,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
        attention_softmax_kernel[(bh*m,)](
            scores,probabilities,N_COLS=n,BLOCK_SIZE=width,
            num_warps=4 if width <= 1024 else 8,
        )
        # In this matmul: output width is dv, and the shared dimension is n.
        attention_value_kernel[(triton.cdiv(m,32),triton.cdiv(dv,32),bh)](
            probabilities,v,output,m,dv,n,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return (output,probabilities) if return_weights else output
```

`bh=batch*heads+head` selects a matrix pair from contiguous rank-four storage. Same-stream launch ordering connects all stages without a CPU synchronization between them. The key-width limit comes from this teaching softmax kernel.

## 6. Match outputs, probability rows, and causal masks

Compare intermediate weights and output values for both attention modes.

```python
for causal in (False,True):
    expected,p_torch = attention_torch(q,k,v,causal,return_weights=True)
    actual,p_triton = attention_triton(q,k,v,causal,return_weights=True)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    torch.testing.assert_close(p_triton,p_torch,rtol=1e-4,atol=1e-6)
    print("causal:",causal,"PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
    print("Maximum output error:",(actual-expected).abs().max().item())
for batches,heads,m,n,d,dv,causal in ((1,1,3,5,7,4,False),(2,2,17,17,33,9,True),(1,2,33,65,31,17,False)):
    queries = torch.randn(batches,heads,m,d,device=DEVICE,dtype=DTYPE)
    keys = torch.randn(batches,heads,n,d,device=DEVICE,dtype=DTYPE)
    values = torch.randn(batches,heads,n,dv,device=DEVICE,dtype=DTYPE)
    actual,weights = attention_triton(queries,keys,values,causal,return_weights=True)
    expected = attention_torch(queries,keys,values,causal)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    torch.testing.assert_close(weights.sum(-1),torch.ones(batches,heads,m,device=DEVICE,dtype=DTYPE),rtol=1e-5,atol=1e-6)
    if causal:
        assert bool(torch.all(weights.triu(diagonal=1) == 0))
print("All attention shape, scale, row-sum, and causal checks passed.")
```

Output tolerances allow FP32 differences in two matmuls and the intervening exponential reductions. The probability checks catch a wrong normalization dimension or masking convention.

## 7. Compare complete attention times

The eager expression and the function use unmasked attention in this timing table.

```python
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("B,H,T,D        eager us  function us  Triton us  speedup  max error")
for batches,heads,length,d in ((1,1,32,32),(1,4,128,64),(2,4,256,64)):
    queries = torch.randn(batches,heads,length,d,device=DEVICE,dtype=DTYPE)
    keys = torch.randn_like(queries)
    values = torch.randn_like(queries)
    expected = attention_torch(queries,keys,values)
    actual = attention_triton(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.softmax((queries @ keys.transpose(-1,-2))/(d**0.5),dim=-1) @ values)
    p = bench_ms(lambda: attention_torch(queries,keys,values))
    t = bench_ms(lambda: attention_triton(queries,keys,values))
    print(f"{batches},{heads},{length},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

Each timed call allocates its output. Input creation, first compilation, assertions, CPU copies, and printing are outside timing. [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock time. A speedup above `1.0x` means the Triton measurement was faster. Run the code for actual timings; no speedup is assumed. The Triton time includes score generation, softmax, value aggregation, and intermediate allocations. Timing only the score kernel would not measure attention.

## 8. Inspect quadratic intermediate storage

Calculate the logical size of the score and probability tensors.

```python
for length in (128,256,512,1024):
    batches,heads = 1,4
    elements = 2 * batches * heads * length * length
    bytes_required = elements * torch.empty((),dtype=DTYPE).element_size()
    print(f"T={length:4d}: scores + probabilities = {bytes_required/2**20:.2f} MiB")
```

Doubling sequence length quadruples this estimate. It counts only the two explicitly retained intermediate tensors, not allocator overhead or peak application memory.

## Try next

Continue with [online softmax](13_online_softmax.md), deriving how the maximum and denominator can be updated together while streaming score blocks.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code: imports, references, kernels, launch helpers, correctness checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and tiny query/key/value tensors
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
q = torch.tensor([[[[1.,0.],[0.,1.]]]],device=DEVICE,dtype=DTYPE)
k = q.clone()
v = torch.tensor([[[[10.,0.],[0.,20.]]]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: expose every attention stage
scores = (q @ k.transpose(-1,-2)) / (q.shape[-1] ** 0.5)
probabilities = torch.softmax(scores,dim=-1)
o_eager = probabilities @ v
print("Scores:\n",scores.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())


# 3. PyTorch function: explicit attention with optional causality
def attention_torch(q,k,v,causal=False,return_weights=False):
    scores = (q @ k.transpose(-1,-2)) / (q.shape[-1] ** 0.5)
    if causal:
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("This example uses square causal self-attention.")
        keep = torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores = scores.masked_fill(~keep,-float("inf"))
    probabilities = torch.softmax(scores,dim=-1)
    output = probabilities @ v
    return (output,probabilities) if return_weights else output


torch.testing.assert_close(attention_torch(q,k,v),o_eager,rtol=0,atol=0)
print("Causal output:\n",attention_torch(q,k,v,causal=True).cpu())


# 4. Triton: three separate kernels
@triton.jit
def attention_scores_kernel(
    Q, Key, Scores, M, N, D, Scale,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten (batch, head) into one independent matrix pair.
    bh = tl.program_id(2)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0,BLOCK_M)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0,BLOCK_N)
    local_d = tl.arange(0,BLOCK_K)
    acc = tl.zeros((BLOCK_M,BLOCK_N),tl.float32)
    for d_tile in range(tl.cdiv(D,BLOCK_K)):
        d = d_tile * BLOCK_K + local_d
        q_ptrs = Q + bh * M * D + rows[:,None] * D + d[None,:]
        # Key is stored as (N,D); this pointer tile presents its transpose.
        k_ptrs = Key + bh * N * D + columns[None,:] * D + d[:,None]
        q = tl.load(q_ptrs,mask=(rows[:,None]<M)&(d[None,:]<D),other=0.0)
        k = tl.load(k_ptrs,mask=(columns[None,:]<N)&(d[:,None]<D),other=0.0)
        acc = tl.dot(q,k,acc,input_precision="ieee")
    scores = acc * Scale
    if CAUSAL:
        scores = tl.where(columns[None,:] <= rows[:,None],scores,-float("inf"))
    tl.store(Scores + bh*M*N + rows[:,None]*N + columns[None,:],scores,
             mask=(rows[:,None]<M)&(columns[None,:]<N))


@triton.jit
def attention_softmax_kernel(X, Y, N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
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



@triton.jit
def attention_value_kernel(
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


# 5. Python wrapper: validate head shapes and allocate intermediates
def attention_triton(q,k,v,causal=False,return_weights=False):
    if not all(t.is_cuda and t.device == q.device for t in (q,k,v)):
        raise ValueError("Inputs must be CUDA tensors on one device.")
    if not all(t.ndim == 4 for t in (q,k,v)):
        raise ValueError("Expected Q/K/V shaped (batch,head,length,features).")
    batches,heads,m,d = q.shape
    n = k.shape[-2]
    dv = v.shape[-1]
    if k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2] or k.shape[-1] != d or v.shape[-2] != n:
        raise ValueError("Batch/head, key/value length, and Q/K feature dimensions must match.")
    if d == 0 or dv == 0 or not 1 <= n <= 4096:
        raise ValueError("Use positive feature dimensions and 1–4096 keys.")
    if causal and m != n:
        raise ValueError("This example uses square causal self-attention.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    bh = batches * heads
    if bh > 65535:
        raise ValueError("This grid-z mapping supports at most 65535 batch/head pairs.")
    scores = torch.empty((batches,heads,m,n),device=q.device,dtype=torch.float32)
    probabilities = torch.empty_like(scores)
    output = torch.empty((batches,heads,m,dv),device=q.device,dtype=torch.float32)
    if bh == 0 or m == 0:
        return (output,probabilities) if return_weights else output
    width = triton.next_power_of_2(n)
    with torch.cuda.device(q.device):
        attention_scores_kernel[(triton.cdiv(m,32),triton.cdiv(n,32),bh)](
            q,k,scores,m,n,d,d**-0.5,CAUSAL=causal,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
        attention_softmax_kernel[(bh*m,)](
            scores,probabilities,N_COLS=n,BLOCK_SIZE=width,
            num_warps=4 if width <= 1024 else 8,
        )
        # In this matmul: output width is dv, and the shared dimension is n.
        attention_value_kernel[(triton.cdiv(m,32),triton.cdiv(dv,32),bh)](
            probabilities,v,output,m,dv,n,
            BLOCK_M=32,BLOCK_N=32,BLOCK_K=32,num_warps=4,
        )
    return (output,probabilities) if return_weights else output


# 6. Match outputs, probability rows, and causal masks
for causal in (False,True):
    expected,p_torch = attention_torch(q,k,v,causal,return_weights=True)
    actual,p_triton = attention_triton(q,k,v,causal,return_weights=True)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    torch.testing.assert_close(p_triton,p_torch,rtol=1e-4,atol=1e-6)
    print("causal:",causal,"PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
    print("Maximum output error:",(actual-expected).abs().max().item())
for batches,heads,m,n,d,dv,causal in ((1,1,3,5,7,4,False),(2,2,17,17,33,9,True),(1,2,33,65,31,17,False)):
    queries = torch.randn(batches,heads,m,d,device=DEVICE,dtype=DTYPE)
    keys = torch.randn(batches,heads,n,d,device=DEVICE,dtype=DTYPE)
    values = torch.randn(batches,heads,n,dv,device=DEVICE,dtype=DTYPE)
    actual,weights = attention_triton(queries,keys,values,causal,return_weights=True)
    expected = attention_torch(queries,keys,values,causal)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    torch.testing.assert_close(weights.sum(-1),torch.ones(batches,heads,m,device=DEVICE,dtype=DTYPE),rtol=1e-5,atol=1e-6)
    if causal:
        assert bool(torch.all(weights.triu(diagonal=1) == 0))
print("All attention shape, scale, row-sum, and causal checks passed.")


# 7. Compare complete attention times
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("B,H,T,D        eager us  function us  Triton us  speedup  max error")
for batches,heads,length,d in ((1,1,32,32),(1,4,128,64),(2,4,256,64)):
    queries = torch.randn(batches,heads,length,d,device=DEVICE,dtype=DTYPE)
    keys = torch.randn_like(queries)
    values = torch.randn_like(queries)
    expected = attention_torch(queries,keys,values)
    actual = attention_triton(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-3)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.softmax((queries @ keys.transpose(-1,-2))/(d**0.5),dim=-1) @ values)
    p = bench_ms(lambda: attention_torch(queries,keys,values))
    t = bench_ms(lambda: attention_triton(queries,keys,values))
    print(f"{batches},{heads},{length},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Inspect quadratic intermediate storage
for length in (128,256,512,1024):
    batches,heads = 1,4
    elements = 2 * batches * heads * length * length
    bytes_required = elements * torch.empty((),dtype=DTYPE).element_size()
    print(f"T={length:4d}: scores + probabilities = {bytes_required/2**20:.2f} MiB")
```
