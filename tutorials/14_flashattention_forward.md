# 14. FlashAttention forward: online normalization and value aggregation

Previous: [online softmax](13_online_softmax.md). Next: [causal FlashAttention](15_causal_flashattention.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 tensors of finite values on one NVIDIA CUDA GPU, with moderate magnitudes whose products and sums remain representable in FP32. Fixed tiles and IEEE dot products make this a teaching implementation of the FlashAttention algorithm; optimized Tensor Core implementations have different performance characteristics. No dropout is used. This guide uses one attention head: Q has shape `(M,D)` and K/V have shape `(N,D)`. The raw Triton forward function does not register a backward operation; guide 17 adds autograd.

## Concepts: compute attention without full score or probability matrices

The mathematical operation remains scaled dot-product attention:

$$
S=QK^T/\sqrt D,\qquad P=\operatorname{softmax}(S),\qquad O=PV.
$$

The dense implementation in [guide 12](12_attention_without_flash.md) stores both $S$ and $P$ in GPU memory. FlashAttention changes how the computation is organized: load a query tile, stream key/value tiles, and consume each temporary score tile immediately. The full $M\times N$ score and probability matrices are never written to global memory in this implementation.

This is an exact-attention algorithm in real arithmetic, rather than a sparse or sampled approximation. Floating-point rounding and exponential approximations can still differ from a dense implementation. See the [FlashAttention paper](https://arxiv.org/abs/2205.14135) for the algorithm's origin and memory-access analysis.

### 1. Online softmax carries a scale and a denominator

For one query row after processing a key prefix $J$, keep

$$
m=\max_{j\in J}s_j,\qquad \ell=\sum_{j\in J}e^{s_j-m}.
$$

A new key block has scores $s_B$. Update the common maximum and rescale old mass:

$$
m'=\max(m,\max s_B),\qquad \alpha=e^{m-m'},\qquad
w_B=e^{s_B-m'},\qquad \ell'=\alpha\ell+\sum_j(w_B)_j.
$$

The maximum can increase, but $\alpha\le1$. Earlier exponentials were measured at the old maximum, so multiplying by $\alpha$ converts them to the new scale. Initialize `m=-inf` and `ell=0`; the first finite block makes the old contribution zero.

### 2. Attention carries a vector numerator at the same scale

Standalone online softmax needs a second pass to write all final probabilities. Attention instead needs only their weighted value sum. Keep a vector

$$
u=\sum_{j\in J}e^{s_j-m}v_j.
$$

When the maximum changes, rescale **both** the denominator and the vector numerator:

$$
\boxed{u'=\alpha u+w_BV_B},\qquad
\boxed{\ell'=\alpha\ell+\sum_j(w_B)_j}.
$$

After the last key block, $o=u/\ell$. Failing to rescale $u$ while rescaling $\ell$ mixes two different exponential scales and changes the attention output. There is no need to write the full probability row or revisit keys merely to normalize its weighted sum.

These invariants hold separately for every query row in the tile. The running maximum and denominator have shape `(BLOCK_M,)`; the numerator has shape `(BLOCK_M,BLOCK_D)`.

```text
keep Q tile locally
       |
for each K/V tile:
    Q @ Kᵀ -> score tile -> row maxima -> new common scale
                                      -> shifted weights
    rescale denominator               -> add row weight sums
    rescale vector numerator          -> add weights @ V
       |
final numerator / denominator -> store O tile
```

### 3. Tiles bound local storage

The teaching kernel uses 32 query rows, 32 keys, and a padded feature tile. Each program retains its query tile and running state while looping over key blocks. Other programs own other query tiles. K/V may be loaded by several programs; the algorithm does not promise a single global read of each input value.

Global output storage is $O(MD)$, with one additional statistic per query row. The temporary score tile is only `32×32`, independent of sequence length. Arithmetic still scales as $O(MND)$; the improvement concerns intermediate storage and data movement, not a change to the attention equation.

### 4. Save log-sum-exp for backward

The kernel also writes the natural-log normalizer

$$
L_i=m_i+\log\ell_i=\log\sum_j e^{S_{i,j}}.
$$

A later backward kernel can reconstruct a probability tile as $P_{i,j}=e^{S_{i,j}-L_i}$. Saving one number per row replaces saving an entire probability row. `LSE` here uses natural logarithms; mixing it with a base-two exponential without converting units would be incorrect.

### 5. Padding and precision

Feature padding loads zero so padded dot-product terms contribute nothing. Key padding sets scores to negative infinity so padded probabilities are zero. Padded query rows need safe statistics: otherwise an all-masked row can evaluate `-inf - (-inf)` and produce NaNs. The kernel sets their temporary maximum to zero and their final divisor to one, and masks their stores.

The scale is $1/\sqrt D$ for the real feature count, not the padded tile width. This guide uses FP32 IEEE dot products and `torch.set_float32_matmul_precision("highest")` to keep the comparison explicit. It demonstrates the algorithm rather than the performance tuning of the [official Triton fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

## 1. Imports and a tiny attention problem

Use a CUDA-enabled PyTorch build and Triton; see the [PyTorch installer](https://pytorch.org/get-started/locally/) and [Triton installation guide](https://triton-lang.org/main/getting-started/installation.html).

```python
import torch
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE=torch.device("cuda",torch.cuda.current_device())
DTYPE=torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6,sci_mode=False)
# Match the teaching kernels' explicit FP32 IEEE dot products.
torch.set_float32_matmul_precision("highest")
print("PyTorch:",torch.__version__,"Triton:",triton.__version__)
print("GPU:",torch.cuda.get_device_name(DEVICE))
q=torch.tensor([[1.,0.],[0.,1.]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[10.,0.],[0.,20.]],device=DEVICE,dtype=DTYPE)
```

The example uses two queries, two keys, and two value features.

## 2. PyTorch eager: inspect the dense reference

Expose the mathematical scores and probabilities.

```python
scores=(q@k.T)*(q.shape[-1]**-0.5)
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
print("Scores:\n",scores.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())
```

The probability rows are approximately `[0.669762,0.330238]` and `[0.330238,0.669762]`. Outputs are approximately `[[6.697615,6.604769],[3.302385,13.395231]]`.

## 3. PyTorch function: reusable dense attention

Keep explicit matmul and softmax calls as the reference.

```python
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


torch.testing.assert_close(attention_torch(q,k,v),o_eager,rtol=0,atol=0)
```

The function can return a dense-reference LSE for testing. It does not call SDPA, which may select a fused backend.

## 4. Triton: stream key/value blocks with online state

Read the two correction updates together.

```python
@triton.jit
def flash_forward_kernel(
    Q, K, V, O, LSE, Scale,
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # One program owns a query tile within one batch/head pair.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, BLOCK_D)
    local_keys = tl.arange(0, BLOCK_N)
    q = tl.load(Q + bh*M*D + rows[:,None]*D + features[None,:],
                mask=(rows[:,None]<M)&(features[None,:]<D), other=0.0)
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    numerator = tl.zeros((BLOCK_M,BLOCK_D), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (query_tile+1)*BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys = start + local_keys
        k_transposed = tl.load(K + bh*N*D + keys[None,:]*D + features[:,None],
                              mask=(keys[None,:]<N)&(features[:,None]<D), other=0.0)
        v = tl.load(V + bh*N*D + keys[:,None]*D + features[None,:],
                    mask=(keys[:,None]<N)&(features[None,:]<D), other=0.0)
        scores = tl.dot(q,k_transposed,input_precision="ieee") * Scale
        allowed = (rows[:,None]<M)&(keys[None,:]<N)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        new_max = tl.maximum(running_max,tl.max(scores,axis=1))
        # Padded query rows must not evaluate -inf - (-inf).
        new_max = tl.where(rows<M,new_max,0.0)
        correction = tl.exp(running_max-new_max)
        weights = tl.exp(scores-new_max[:,None])
        running_sum = correction*running_sum + tl.sum(weights,axis=1)
        # Rescale the vector numerator as well as the scalar denominator.
        numerator = numerator * correction[:,None]
        numerator = tl.dot(weights,v,numerator,input_precision="ieee")
        running_max = new_max
    safe_sum = tl.where(rows<M,running_sum,1.0)
    output = numerator / safe_sum[:,None]
    tl.store(O + bh*M*D + rows[:,None]*D + features[None,:],output,
             mask=(rows[:,None]<M)&(features[None,:]<D))
    # Natural-log LSE is enough to reconstruct probabilities in backward.
    tl.store(LSE + bh*M + rows,running_max+tl.log(safe_sum),mask=rows<M)
```

1. Load and retain one query tile.
2. Compute each key block's score tile and its row maxima.
3. Correct both old denominator and old numerator to the new scale.
4. Add the block's denominator mass and weighted values.
5. Divide once at the end and store output plus row LSE.

The causal specialization is included in the kernel for the next guide; this wrapper launches it with `CAUSAL=False`.

## 5. Python wrapper: allocate output and row statistics

No full score or probability tensor is allocated.

```python
def flash_forward_triton(q,k,v,return_lse=False):
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim==2 for t in (q,k,v)) or k.shape!=v.shape or q.shape[1]!=k.shape[1]:
        raise ValueError("Use Q=(M,D), K=V=(N,D).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 matrices.")
    m,d=q.shape
    n=k.shape[0]
    if not 0<=m<=4096 or not 1<=n<=4096 or not 1<=d<=128:
        raise ValueError("Use M=0–4096, N=1–4096, and D=1–128.")
    out=torch.empty_like(q)
    lse=torch.empty(m,device=q.device,dtype=torch.float32)
    if m:
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(m,32),1)](
                q,k,v,out,lse,d**-0.5,M=m,N=n,D=d,CAUSAL=False,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=max(32,triton.next_power_of_2(d)),num_warps=4,
            )
    return (out,lse) if return_lse else out
```

The positive key-count check guarantees at least one valid score per real query. The width limits bound this teaching example, not the mathematical algorithm.

## 6. Match values, LSE, and partial tiles

Check both the consumer output and the saved normalizer.

```python
actual,lse=flash_forward_triton(q,k,v,return_lse=True)
expected,ref_lse=attention_torch(q,k,v,return_lse=True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for m,n,d in ((0,7,3),(1,1,1),(17,33,7),(33,65,31),(65,17,64),(5,7,128)):
    queries=torch.randn(m,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn(n,d,device=DEVICE,dtype=DTYPE)
    values=torch.randn_like(keys)
    out,saved=flash_forward_triton(queries,keys,values,True)
    ref,normalizer=attention_torch(queries,keys,values,return_lse=True)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
print("All forward, cross-attention, padded-row, and LSE checks passed.")
```

Different query/key lengths test unmasked cross-attention. Irregular feature counts check zero padding and use of the real scaling dimension.

## 7. Compare eager, function, and fused forward times

Measure completed attention, including the saved LSE buffer.

```python
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("M,N,D       eager us  function us  Triton us  speedup  max error")
for m,n,d in ((32,32,32),(128,128,64),(256,256,64),(512,512,64)):
    queries=torch.randn(m,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn(n,d,device=DEVICE,dtype=DTYPE)
    values=torch.randn_like(keys)
    actual=flash_forward_triton(queries,keys,values)
    expected=attention_torch(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    error=(actual-expected).abs().max().item()
    e=bench_ms(lambda:torch.softmax((queries@keys.T)*(d**-0.5),dim=-1)@values)
    p=bench_ms(lambda:attention_torch(queries,keys,values))
    t=bench_ms(lambda:flash_forward_triton(queries,keys,values))
    print(f"{m},{n},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. Input creation, first compilation, correctness checks, CPU copies, and printing are outside timing. Each callable allocates its outputs. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock measurements. Speedup is PyTorch function time divided by Triton time. Run the code to get actual measurements; no speedup is assumed. The baseline is explicit dense PyTorch attention. This comparison is not a benchmark against PyTorch's optimized FlashAttention backend.

## 8. Trace denominator and numerator rescaling

A four-score example shows why both corrections are necessary.

```python
score_row=torch.tensor([1.,2.,3.,4.],device=DEVICE,dtype=DTYPE)
value_rows=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[0.,2.]],device=DEVICE,dtype=DTYPE)
maximum=torch.tensor(-float("inf"),device=DEVICE,dtype=DTYPE)
denominator=torch.zeros((),device=DEVICE,dtype=DTYPE)
numerator=torch.zeros(2,device=DEVICE,dtype=DTYPE)
for start in (0,2):
    block=score_row[start:start+2]
    new_maximum=torch.maximum(maximum,block.max())
    correction=torch.exp(maximum-new_maximum)
    weights=torch.exp(block-new_maximum)
    denominator=denominator*correction+weights.sum()
    numerator=numerator*correction+weights@value_rows[start:start+2]
    maximum=new_maximum
    print("max:",maximum.item(),"denominator:",denominator.item(),"numerator:",numerator.cpu().tolist())
result=numerator/denominator
reference=torch.softmax(score_row,dim=-1)@value_rows
torch.testing.assert_close(result,reference,rtol=1e-5,atol=1e-6)
print("Final weighted value:",result.cpu().tolist())
```

The second block raises the maximum from 2 to 4. Rescaling the old numerator by `exp(-2)` preserves the same attention equation without a second key scan.

## Try next

Continue with [causal FlashAttention](15_causal_flashattention.md), adding triangular masks and skipping key tiles that are entirely in the future.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code, including imports, PyTorch references, Triton kernels, wrappers, checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and a tiny attention problem
import torch
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE=torch.device("cuda",torch.cuda.current_device())
DTYPE=torch.float32
torch.manual_seed(0)
torch.set_printoptions(precision=6,sci_mode=False)
# Match the teaching kernels' explicit FP32 IEEE dot products.
torch.set_float32_matmul_precision("highest")
print("PyTorch:",torch.__version__,"Triton:",triton.__version__)
print("GPU:",torch.cuda.get_device_name(DEVICE))
q=torch.tensor([[1.,0.],[0.,1.]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[10.,0.],[0.,20.]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: inspect the dense reference
scores=(q@k.T)*(q.shape[-1]**-0.5)
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
print("Scores:\n",scores.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())


# 3. PyTorch function: reusable dense attention
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


torch.testing.assert_close(attention_torch(q,k,v),o_eager,rtol=0,atol=0)


# 4. Triton: stream key/value blocks with online state
@triton.jit
def flash_forward_kernel(
    Q, K, V, O, LSE, Scale,
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # One program owns a query tile within one batch/head pair.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    features = tl.arange(0, BLOCK_D)
    local_keys = tl.arange(0, BLOCK_N)
    q = tl.load(Q + bh*M*D + rows[:,None]*D + features[None,:],
                mask=(rows[:,None]<M)&(features[None,:]<D), other=0.0)
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    numerator = tl.zeros((BLOCK_M,BLOCK_D), tl.float32)
    end = N
    if CAUSAL:
        end = tl.minimum(N, (query_tile+1)*BLOCK_M)
    for start in range(0, end, BLOCK_N):
        keys = start + local_keys
        k_transposed = tl.load(K + bh*N*D + keys[None,:]*D + features[:,None],
                              mask=(keys[None,:]<N)&(features[:,None]<D), other=0.0)
        v = tl.load(V + bh*N*D + keys[:,None]*D + features[None,:],
                    mask=(keys[:,None]<N)&(features[None,:]<D), other=0.0)
        scores = tl.dot(q,k_transposed,input_precision="ieee") * Scale
        allowed = (rows[:,None]<M)&(keys[None,:]<N)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        new_max = tl.maximum(running_max,tl.max(scores,axis=1))
        # Padded query rows must not evaluate -inf - (-inf).
        new_max = tl.where(rows<M,new_max,0.0)
        correction = tl.exp(running_max-new_max)
        weights = tl.exp(scores-new_max[:,None])
        running_sum = correction*running_sum + tl.sum(weights,axis=1)
        # Rescale the vector numerator as well as the scalar denominator.
        numerator = numerator * correction[:,None]
        numerator = tl.dot(weights,v,numerator,input_precision="ieee")
        running_max = new_max
    safe_sum = tl.where(rows<M,running_sum,1.0)
    output = numerator / safe_sum[:,None]
    tl.store(O + bh*M*D + rows[:,None]*D + features[None,:],output,
             mask=(rows[:,None]<M)&(features[None,:]<D))
    # Natural-log LSE is enough to reconstruct probabilities in backward.
    tl.store(LSE + bh*M + rows,running_max+tl.log(safe_sum),mask=rows<M)


# 5. Python wrapper: allocate output and row statistics
def flash_forward_triton(q,k,v,return_lse=False):
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim==2 for t in (q,k,v)) or k.shape!=v.shape or q.shape[1]!=k.shape[1]:
        raise ValueError("Use Q=(M,D), K=V=(N,D).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 matrices.")
    m,d=q.shape
    n=k.shape[0]
    if not 0<=m<=4096 or not 1<=n<=4096 or not 1<=d<=128:
        raise ValueError("Use M=0–4096, N=1–4096, and D=1–128.")
    out=torch.empty_like(q)
    lse=torch.empty(m,device=q.device,dtype=torch.float32)
    if m:
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(m,32),1)](
                q,k,v,out,lse,d**-0.5,M=m,N=n,D=d,CAUSAL=False,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=max(32,triton.next_power_of_2(d)),num_warps=4,
            )
    return (out,lse) if return_lse else out


# 6. Match values, LSE, and partial tiles
actual,lse=flash_forward_triton(q,k,v,return_lse=True)
expected,ref_lse=attention_torch(q,k,v,return_lse=True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for m,n,d in ((0,7,3),(1,1,1),(17,33,7),(33,65,31),(65,17,64),(5,7,128)):
    queries=torch.randn(m,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn(n,d,device=DEVICE,dtype=DTYPE)
    values=torch.randn_like(keys)
    out,saved=flash_forward_triton(queries,keys,values,True)
    ref,normalizer=attention_torch(queries,keys,values,return_lse=True)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
print("All forward, cross-attention, padded-row, and LSE checks passed.")


# 7. Compare eager, function, and fused forward times
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("M,N,D       eager us  function us  Triton us  speedup  max error")
for m,n,d in ((32,32,32),(128,128,64),(256,256,64),(512,512,64)):
    queries=torch.randn(m,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn(n,d,device=DEVICE,dtype=DTYPE)
    values=torch.randn_like(keys)
    actual=flash_forward_triton(queries,keys,values)
    expected=attention_torch(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    error=(actual-expected).abs().max().item()
    e=bench_ms(lambda:torch.softmax((queries@keys.T)*(d**-0.5),dim=-1)@values)
    p=bench_ms(lambda:attention_torch(queries,keys,values))
    t=bench_ms(lambda:flash_forward_triton(queries,keys,values))
    print(f"{m},{n},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Trace denominator and numerator rescaling
score_row=torch.tensor([1.,2.,3.,4.],device=DEVICE,dtype=DTYPE)
value_rows=torch.tensor([[1.,0.],[0.,1.],[2.,0.],[0.,2.]],device=DEVICE,dtype=DTYPE)
maximum=torch.tensor(-float("inf"),device=DEVICE,dtype=DTYPE)
denominator=torch.zeros((),device=DEVICE,dtype=DTYPE)
numerator=torch.zeros(2,device=DEVICE,dtype=DTYPE)
for start in (0,2):
    block=score_row[start:start+2]
    new_maximum=torch.maximum(maximum,block.max())
    correction=torch.exp(maximum-new_maximum)
    weights=torch.exp(block-new_maximum)
    denominator=denominator*correction+weights.sum()
    numerator=numerator*correction+weights@value_rows[start:start+2]
    maximum=new_maximum
    print("max:",maximum.item(),"denominator:",denominator.item(),"numerator:",numerator.cpu().tolist())
result=numerator/denominator
reference=torch.softmax(score_row,dim=-1)@value_rows
torch.testing.assert_close(result,reference,rtol=1e-5,atol=1e-6)
print("Final weighted value:",result.cpu().tolist())
```
