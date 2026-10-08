# 15. Causal FlashAttention: triangular scores with online normalization

Previous: [FlashAttention forward](14_flashattention_forward.md). Next: [multi-head FlashAttention](16_multi_head_flashattention.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 tensors of finite values on one NVIDIA CUDA GPU, with moderate magnitudes whose products and sums remain representable in FP32. Fixed tiles and IEEE dot products make this a teaching implementation of the FlashAttention algorithm; optimized Tensor Core implementations have different performance characteristics. No dropout is used. Q/K/V have matching rank-two shapes `(T,D)`, with T=1–4096 and D=1–128. This guide implements square causal self-attention and has no registered backward operation.

## Concepts: future positions receive zero weight

In an autoregressive sequence, query position $i$ may use keys $j\le i$. The score and output equations are

$$
S_{i,j}=\begin{cases}q_i\cdot k_j/\sqrt D,&j\le i,\\-\infty,&j>i,\end{cases}
\qquad P_{i,j}=\operatorname{softmax}(S_i)_j,\qquad o_i=\sum_{j\le i}P_{i,j}v_j.
$$

Mask **before** exponentiation and normalization. Multiplying future probabilities by zero afterward would leave the denominator counting forbidden positions. Each real query includes key zero and its own position, so its allowed score set is nonempty.

```text
query/key        0  1  2  3
0                A  -  -  -
1                A  A  -  -
2                A  A  A  -
3                A  A  A  A
A = allowed score; - = negative infinity
```

### 1. Online statistics use only allowed scores

For the processed allowed key prefix, maintain a row maximum $m_i$, shifted denominator $\ell_i$, and value numerator $u_i$. For each tile, set forbidden scores to negative infinity, then use

$$
m_i'=\max(m_i,\max S_{i,B}),\quad \alpha_i=e^{m_i-m_i'},\quad W_{i,B}=e^{S_{i,B}-m_i'},
$$

$$
\ell_i'=\alpha_i\ell_i+\sum_j W_{i,j},\qquad
u_i'=\alpha_i u_i+W_{i,B}V_B.
$$

A future entry contributes zero to both sums. For an already initialized real query whose entire current block is future-masked, its old maximum stays finite, the correction is one, and the block adds no mass. The numerator and denominator still share the same scale.

### 2. Past, diagonal, and future tiles

A key tile entirely before the query tile contains allowed positions. A diagonal tile may contain both allowed and forbidden positions and needs an elementwise `key <= query` mask. A tile entirely after the last query in the program can be skipped.

For a query tile starting at `query_tile*BLOCK_M`, this teaching kernel sets its key-loop end to `min(T,(query_tile+1)*BLOCK_M)`. That excludes fully future key tiles. The final visited tile still gets the triangular mask; a tile boundary does not replace the per-element causal test.

```text
query block's key scan:
[fully past tiles] [diagonal overlap] [fully future tiles]
      visit              visit              skip
```

The code applies the triangular mask on every visited tile for clarity. A tuned implementation can use separate fully-past and diagonal paths.

### 3. Causal masks and padding are different conditions

A key can be within the sequence but causally forbidden. A padded key can be outside the sequence entirely. Both should have zero exponential weight, but only the latter must also be masked on memory loads. Padded query rows must not create invalid `-inf - (-inf)` statistics; the kernel gives them safe temporary state and masks their output and LSE stores.

### 4. Mask alignment is part of the operation

This tutorial assumes Q and K refer to the same length-T sequence with positions starting at zero. Rectangular cross-attention and decoding with a cached key prefix need explicitly aligned position indices; simply applying `key <= local_query_index` can then express the wrong mask. The examples stay with square self-attention so the convention is unambiguous.

The algorithm still stores only O and one natural-log LSE value per query row. It does not allocate a triangular probability matrix. Causal attention reduces the allowed score pairs to $T(T+1)/2$, although this simple tiled implementation also performs some masked products in diagonal tiles. The [official fused-attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) includes more advanced scheduling.

## 1. Imports and a two-position example

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

Query zero can use only value zero; query one can use both values.

## 2. PyTorch eager: mask scores before softmax

Construct the triangular mask explicitly.

```python
scores=(q@k.T)*(q.shape[-1]**-0.5)
allowed=torch.ones(q.shape[0],k.shape[0],device=DEVICE,dtype=torch.bool).tril()
masked=scores.masked_fill(~allowed,-float("inf"))
probabilities=torch.softmax(masked,dim=-1)
o_eager=probabilities@v
print("Masked scores:\n",masked.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())
```

The first probability row is `[1,0]`, so its output is exactly `[10,0]`. The second output is approximately `[3.302385,13.395231]`.

## 3. PyTorch function: square causal attention

Use the same explicit dense reference with the causal flag.

```python
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


torch.testing.assert_close(attention_torch(q,k,v,causal=True),o_eager,rtol=0,atol=0)
```

The triangular mask broadcasts across any leading dimensions, although this guide's public Triton wrapper uses one head.

## 4. Triton: skip future tiles and mask diagonal scores

The causal specialization modifies the loop bound and score-validity condition.

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

Read the `if CAUSAL` statements together: one limits visited key blocks, and the other masks individual scores. The same online correction rescales denominator and numerator for each query row.

## 5. Python wrapper: enforce square causal shapes

Launch only the causal specialization.

```python
def causal_flash_triton(q,k,v,return_lse=False):
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim==2 for t in (q,k,v)) or k.shape!=v.shape or q.shape[1]!=k.shape[1]:
        raise ValueError("Use Q=(M,D), K=V=(N,D).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 matrices.")
    if q.shape!=k.shape:
        raise ValueError("Causal self-attention requires matching Q/K/V shapes.")
    m,d=q.shape
    n=k.shape[0]
    if not 0<=m<=4096 or not 1<=n<=4096 or not 1<=d<=128:
        raise ValueError("Use M=0–4096, N=1–4096, and D=1–128.")
    out=torch.empty_like(q)
    lse=torch.empty(m,device=q.device,dtype=torch.float32)
    if m:
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(m,32),1)](
                q,k,v,out,lse,d**-0.5,M=m,N=n,D=d,CAUSAL=True,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=max(32,triton.next_power_of_2(d)),num_warps=4,
            )
    return (out,lse) if return_lse else out
```

The wrapper requires equal Q/K/V shapes so that the simple position comparison describes the intended self-attention mask.

## 6. Compare outputs, LSE, and causal edge cases

Sequence length one and partial diagonal tiles are useful checks.

```python
actual,lse=causal_flash_triton(q,k,v,True)
expected,ref_lse=attention_torch(q,k,v,True,True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for length,d in ((1,1),(17,7),(33,31),(65,64),(5,128)):
    queries=torch.randn(length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    out,saved=causal_flash_triton(queries,keys,values,True)
    ref,normalizer=attention_torch(queries,keys,values,True,True)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(out[0],values[0],rtol=1e-5,atol=1e-6)
print("All causal mask, diagonal-tail, and LSE checks passed.")
```

At length one, attention returns its only value. The last query can see the whole sequence; earlier queries use shorter prefixes.

## 7. Compare complete causal forward times

Prepare one reusable mask for the direct eager timing.

```python
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("T,D      eager us  function us  Triton us  speedup  max error")
for length,d in ((32,32),(128,64),(256,64),(512,64)):
    queries=torch.randn(length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    mask=torch.ones(length,length,device=DEVICE,dtype=torch.bool).tril()
    expected=attention_torch(queries,keys,values,True)
    actual=causal_flash_triton(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    error=(actual-expected).abs().max().item()
    e=bench_ms(lambda:torch.softmax(((queries@keys.T)*(d**-0.5)).masked_fill(~mask,-float("inf")),dim=-1)@values)
    p=bench_ms(lambda:attention_torch(queries,keys,values,True))
    t=bench_ms(lambda:causal_flash_triton(queries,keys,values))
    print(f"{length},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. Input creation, first compilation, correctness checks, CPU copies, and printing are outside timing. Each callable allocates its outputs. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock measurements. Speedup is PyTorch function time divided by Triton time. Run the code to get actual measurements; no speedup is assumed. The direct eager callable reuses a mask; the reusable PyTorch function constructs its mask each call. Speedup uses the function time, so retain that distinction when interpreting the table.

## 8. Verify that future values cannot change earlier outputs

Perturb only the final position's value.

```python
queries=torch.randn(33,7,device=DEVICE,dtype=DTYPE)
keys=torch.randn_like(queries)
values=torch.randn_like(queries)
original=causal_flash_triton(queries,keys,values)
changed_values=values.clone()
changed_values[-1]+=10.0
changed=causal_flash_triton(queries,keys,changed_values)
torch.testing.assert_close(changed[:-1],original[:-1],rtol=0,atol=0)
print("Earlier outputs were unaffected by the future-value change.")
```

This checks causality through output behavior rather than only checking the mask tensor.

## Try next

Continue with [multi-head FlashAttention](16_multi_head_flashattention.md), adding independent batch/head offsets to the same tiled algorithm.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code, including imports, PyTorch references, Triton kernels, wrappers, checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and a two-position example
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


# 2. PyTorch eager: mask scores before softmax
scores=(q@k.T)*(q.shape[-1]**-0.5)
allowed=torch.ones(q.shape[0],k.shape[0],device=DEVICE,dtype=torch.bool).tril()
masked=scores.masked_fill(~allowed,-float("inf"))
probabilities=torch.softmax(masked,dim=-1)
o_eager=probabilities@v
print("Masked scores:\n",masked.cpu())
print("Probabilities:\n",probabilities.cpu())
print("Output:\n",o_eager.cpu())


# 3. PyTorch function: square causal attention
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


torch.testing.assert_close(attention_torch(q,k,v,causal=True),o_eager,rtol=0,atol=0)


# 4. Triton: skip future tiles and mask diagonal scores
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


# 5. Python wrapper: enforce square causal shapes
def causal_flash_triton(q,k,v,return_lse=False):
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim==2 for t in (q,k,v)) or k.shape!=v.shape or q.shape[1]!=k.shape[1]:
        raise ValueError("Use Q=(M,D), K=V=(N,D).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 matrices.")
    if q.shape!=k.shape:
        raise ValueError("Causal self-attention requires matching Q/K/V shapes.")
    m,d=q.shape
    n=k.shape[0]
    if not 0<=m<=4096 or not 1<=n<=4096 or not 1<=d<=128:
        raise ValueError("Use M=0–4096, N=1–4096, and D=1–128.")
    out=torch.empty_like(q)
    lse=torch.empty(m,device=q.device,dtype=torch.float32)
    if m:
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(m,32),1)](
                q,k,v,out,lse,d**-0.5,M=m,N=n,D=d,CAUSAL=True,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=max(32,triton.next_power_of_2(d)),num_warps=4,
            )
    return (out,lse) if return_lse else out


# 6. Compare outputs, LSE, and causal edge cases
actual,lse=causal_flash_triton(q,k,v,True)
expected,ref_lse=attention_torch(q,k,v,True,True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for length,d in ((1,1),(17,7),(33,31),(65,64),(5,128)):
    queries=torch.randn(length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    out,saved=causal_flash_triton(queries,keys,values,True)
    ref,normalizer=attention_torch(queries,keys,values,True,True)
    torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(out[0],values[0],rtol=1e-5,atol=1e-6)
print("All causal mask, diagonal-tail, and LSE checks passed.")


# 7. Compare complete causal forward times
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("T,D      eager us  function us  Triton us  speedup  max error")
for length,d in ((32,32),(128,64),(256,64),(512,64)):
    queries=torch.randn(length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    mask=torch.ones(length,length,device=DEVICE,dtype=torch.bool).tril()
    expected=attention_torch(queries,keys,values,True)
    actual=causal_flash_triton(queries,keys,values)
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    error=(actual-expected).abs().max().item()
    e=bench_ms(lambda:torch.softmax(((queries@keys.T)*(d**-0.5)).masked_fill(~mask,-float("inf")),dim=-1)@values)
    p=bench_ms(lambda:attention_torch(queries,keys,values,True))
    t=bench_ms(lambda:causal_flash_triton(queries,keys,values))
    print(f"{length},{d} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Verify that future values cannot change earlier outputs
queries=torch.randn(33,7,device=DEVICE,dtype=DTYPE)
keys=torch.randn_like(queries)
values=torch.randn_like(queries)
original=causal_flash_triton(queries,keys,values)
changed_values=values.clone()
changed_values[-1]+=10.0
changed=causal_flash_triton(queries,keys,changed_values)
torch.testing.assert_close(changed[:-1],original[:-1],rtol=0,atol=0)
print("Earlier outputs were unaffected by the future-value change.")
```
