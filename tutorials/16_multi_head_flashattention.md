# 16. Multi-head FlashAttention: independent online state for every head

Previous: [causal FlashAttention](15_causal_flashattention.md). Next: [FlashAttention backward](17_flashattention_backward.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 tensors of finite values on one NVIDIA CUDA GPU, with moderate magnitudes whose products and sums remain representable in FP32. Fixed tiles and IEEE dot products make this a teaching implementation of the FlashAttention algorithm; optimized Tensor Core implementations have different performance characteristics. No dropout is used. Q/K/V have matching shapes `(B,H,T,D)`. Both unmasked and square causal self-attention are supported. The raw forward wrapper has no registered backward operation; guide 17 supplies it.

## Concepts: many heads, separate attention distributions

Multi-head attention runs several attention problems in parallel. Each head has its own queries, keys, values, score rows, and normalization. For batch $b$ and head $h$,

$$
S_{b,h,i,j}=\frac{q_{b,h,i}\cdot k_{b,h,j}}{\sqrt D},\qquad
P_{b,h,i,:}=\operatorname{softmax}(S_{b,h,i,:}),\qquad
O_{b,h,i,:}=\sum_j P_{b,h,i,j}v_{b,h,j,:}.
$$

Here $D$ is the feature count **within one head**. Scaling by $1/\sqrt{HD}$ would change the operation. Softmax runs along keys, never along the head axis. Causal mode additionally assigns negative infinity to $j>i$ within each head.

### 1. Projection, attention, and head merging are separate steps

A transformer commonly projects an input with embedding width $E$ into Q/K/V, reshapes those projections into heads, performs attention, and then applies an output projection. This notebook starts with already projected, contiguous Q/K/V tensors. Its kernel computes attention only; it does not include projection weights or positional encodings.

```text
already projected Q/K/V, shape (B,H,T,D)
       |                  |                  |
    head 0             head 1             head H-1
  own scores         own scores          own scores
  own m,l,u          own m,l,u           own m,l,u
       |                  |                  |
       +---- outputs (B,H,T,D) ------------+
                         |
             transpose -> concatenate heads
                         |
                  shape (B,T,H*D)
```

After attention, concatenating head features uses `out.transpose(1,2).contiguous().reshape(B,T,H*D)`. The explicit contiguous copy gives the requested memory order after transposing. An optional output projection would then multiply by a matrix of shape `(H*D,E)`; it is outside the timed attention call.

### 2. Each query in each head owns an online state

For a key prefix in a fixed batch/head pair, retain

$$
m_i=\max_j S_{i,j},\quad \ell_i=\sum_j e^{S_{i,j}-m_i},\quad
u_i=\sum_j e^{S_{i,j}-m_i}v_j.
$$

The block update is the same one derived in [guide 14](14_flashattention_forward.md):

$$
m_i'=\max(m_i,\max S_{i,B}),\quad \alpha_i=e^{m_i-m_i'},\quad W_{i,B}=e^{S_{i,B}-m_i'},
$$

$$
\ell_i'=\alpha_i\ell_i+\sum_j W_{i,j},\qquad
u_i'=\alpha_i u_i+W_{i,B}V_B,\qquad o_i=u_i/\ell_i.
$$

The numerator and denominator must be rescaled together. Heads never share these statistics. Saving $L_i=m_i+\log\ell_i$ produces an LSE tensor of shape `(B,H,T)` for later backward reconstruction.

### 3. Contiguous layout determines the pointer arithmetic

For `(B,H,T,D)` in contiguous order, feature coordinates are adjacent, then sequence positions, then heads, then batches. Flatten the batch/head pair as `bh=b*H+h`. Its Q/K/V/O base offset is `bh*T*D`, and a row/feature address adds `row*D+feature`. The LSE base offset is `bh*T`.

The launch grid has two dimensions:

```text
program_id(0): query tile, 0 ... ceil(T/32)-1
program_id(1): batch/head pair, 0 ... B*H-1
```

Each program stores a disjoint query tile within one head. No communication between heads is needed. A tensor produced by `transpose` may have the right shape but different strides; this wrapper requires `.contiguous()` before launch. Arbitrary-stride support would need explicit stride arguments and corresponding pointer calculations.

### 4. More heads add parallel work and storage

Dense attention would store probability tensors with $BHT^2$ elements. This kernel stores O with $BHTD$ elements and LSE with $BHT$ elements, retaining only bounded score tiles locally. Total arithmetic is still proportional to $BHT^2D$. Head independence provides more programs, but speed depends on shapes, hardware, occupancy, and tile choices; extra heads do not guarantee a particular speedup.

The wrapper limits T to 1–4096, D to 1–128, and B*H to at most 65535 for this launch layout. A zero batch or head count returns empty tensors without launching kernels. Every nonempty sequence has at least one allowed key per query. The [official Triton attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) provides a more heavily tuned implementation.

## 1. Imports and two distinct heads

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
q=torch.tensor([[[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[[[10.,0.],[0.,20.]],[[1.,2.],[3.,4.]]]],device=DEVICE,dtype=DTYPE)
```

The two heads use different values so accidental mixing is visible in the output.

## 2. PyTorch eager: broadcast matmul across batch and heads

The last two axes form each independent matrix product.

```python
scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
print("Probability shape:",tuple(probabilities.shape))
print("Head 0 output:\n",o_eager[0,0].cpu())
print("Head 1 output:\n",o_eager[0,1].cpu())
```

The probability tensor has shape `(1,2,2,2)`. Head 0 matches guide 14; head 1 gives approximately `[[1.660477,2.660477],[2.339523,3.339523]]`.

## 3. PyTorch function: optional causal masking per head

The same two-dimensional triangle broadcasts over B and H.

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

This remains a dense, explicit PyTorch reference for both causal modes.

## 4. Triton: add batch/head offsets to every pointer

The numerical state and update remain local to one head.

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

1. Use the second program ID to select a batch/head base address.
2. Load one query tile from that head.
3. Stream only that head's K/V tiles and update its row statistics.
4. Apply `key <= query` if the causal specialization is selected.
5. Store the head's output and LSE at matching offsets.

## 5. Python wrapper: launch the two-dimensional grid

Allocate O and LSE in the same batch/head order.

```python
def flash_forward_triton(q,k,v,causal=False,return_lse=False):
    if not all(t.is_cuda and t.device == q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim == 4 and t.shape == q.shape for t in (q,k,v)):
        raise ValueError("Use matching (batch,head,length,features) tensors.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    batches,heads,length,d = q.shape
    if not 1 <= length <= 4096 or not 1 <= d <= 128 or batches*heads > 65535:
        raise ValueError("Use T=1–4096, D=1–128, and at most 65535 batch/head pairs.")
    out = torch.empty_like(q)
    lse = torch.empty((batches,heads,length),device=q.device,dtype=torch.float32)
    if batches*heads:
        block_d = max(32,triton.next_power_of_2(d))
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(length,32),batches*heads)](
                q,k,v,out,lse,d**-0.5,M=length,N=length,D=d,CAUSAL=causal,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4,
            )
    return (out,lse) if return_lse else out
```

The feature padding is local to dot products. Stored output features remain exactly D wide, and the scaling factor also uses D.

## 6. Match both causal modes, multiple heads, and tail tiles

Test empty batches as well as several nonempty layouts.

```python
actual,lse=flash_forward_triton(q,k,v,return_lse=True)
expected,ref_lse=attention_torch(q,k,v,return_lse=True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for b,h,length,d in ((0,2,3,7),(1,1,1,1),(2,3,17,7),(1,2,33,31),(1,2,65,64),(1,1,5,128)):
    queries=torch.randn(b,h,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    for causal in (False,True):
        out,saved=flash_forward_triton(queries,keys,values,causal,True)
        ref,normalizer=attention_torch(queries,keys,values,causal,True)
        torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
print("All batch/head, causal, empty-batch, and padding checks passed.")
```

These checks exercise the flattened head offsets rather than only testing a single head.

## 7. Compare eager, function, and Triton multi-head times

Time attention in both causal modes; head merging is outside timing.

```python
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("B,H,T,D       causal  eager us  function us  Triton us  speedup  max error")
for b,h,length,d in ((1,1,32,32),(1,4,128,64),(2,4,256,64)):
    queries=torch.randn(b,h,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    mask=torch.ones(length,length,device=DEVICE,dtype=torch.bool).tril()
    for causal in (False,True):
        # Direct eager operators, with a reusable mask for causal mode.
        def eager_call():
            s=(queries@keys.transpose(-1,-2))*(d**-0.5)
            if causal:
                s=s.masked_fill(~mask,-float("inf"))
            return torch.softmax(s,dim=-1)@values
        expected=attention_torch(queries,keys,values,causal)
        actual=flash_forward_triton(queries,keys,values,causal)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        error=(actual-expected).abs().max().item()
        e=bench_ms(eager_call)
        p=bench_ms(lambda:attention_torch(queries,keys,values,causal))
        t=bench_ms(lambda:flash_forward_triton(queries,keys,values,causal))
        print(f"{b},{h},{length},{d} {str(causal):>7} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")
```

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. Input creation, first compilation, correctness checks, CPU copies, and printing are outside timing. Each callable allocates its outputs. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock measurements. Speedup is PyTorch function time divided by Triton time. Run the code to get actual measurements; no speedup is assumed. As in guide 15, direct eager causal timing reuses its mask while the reusable reference constructs one each call. No projection or head-concatenation work is included.

## 8. Check head independence and concatenate head features

Change one head's V without changing any other head.

```python
original=flash_forward_triton(q,k,v)
changed_values=v.clone()
changed_values[:,0]+=5.0
changed=flash_forward_triton(q,k,changed_values)
torch.testing.assert_close(changed[:,1:],original[:,1:],rtol=0,atol=0)
torch.testing.assert_close(changed[:,0],original[:,0]+5.0,rtol=1e-5,atol=1e-5)
b,h,length,d=original.shape
merged=original.transpose(1,2).contiguous().reshape(b,length,h*d)
print("Independent heads passed; concatenated shape:",tuple(merged.shape))
print("Concatenated output:\n",merged.cpu())
```

Adding the same vector to every value in one head adds that vector to its output because its probabilities sum to one. The other head stays unchanged.

## Try next

Continue with [FlashAttention backward](17_flashattention_backward.md), reconstructing probability tiles and registering gradients with PyTorch autograd.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code, including imports, PyTorch references, Triton kernels, wrappers, checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and two distinct heads
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
q=torch.tensor([[[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[[[10.,0.],[0.,20.]],[[1.,2.],[3.,4.]]]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: broadcast matmul across batch and heads
scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
print("Probability shape:",tuple(probabilities.shape))
print("Head 0 output:\n",o_eager[0,0].cpu())
print("Head 1 output:\n",o_eager[0,1].cpu())


# 3. PyTorch function: optional causal masking per head
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


torch.testing.assert_close(attention_torch(q,k,v),o_eager,rtol=0,atol=0)


# 4. Triton: add batch/head offsets to every pointer
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


# 5. Python wrapper: launch the two-dimensional grid
def flash_forward_triton(q,k,v,causal=False,return_lse=False):
    if not all(t.is_cuda and t.device == q.device for t in (q,k,v)):
        raise ValueError("Q/K/V must be CUDA tensors on the same device.")
    if not all(t.ndim == 4 and t.shape == q.shape for t in (q,k,v)):
        raise ValueError("Use matching (batch,head,length,features) tensors.")
    if not all(t.dtype == torch.float32 and t.is_contiguous() for t in (q,k,v)):
        raise ValueError("Inputs must be contiguous FP32 tensors.")
    batches,heads,length,d = q.shape
    if not 1 <= length <= 4096 or not 1 <= d <= 128 or batches*heads > 65535:
        raise ValueError("Use T=1–4096, D=1–128, and at most 65535 batch/head pairs.")
    out = torch.empty_like(q)
    lse = torch.empty((batches,heads,length),device=q.device,dtype=torch.float32)
    if batches*heads:
        block_d = max(32,triton.next_power_of_2(d))
        with torch.cuda.device(q.device):
            flash_forward_kernel[(triton.cdiv(length,32),batches*heads)](
                q,k,v,out,lse,d**-0.5,M=length,N=length,D=d,CAUSAL=causal,
                BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4,
            )
    return (out,lse) if return_lse else out


# 6. Match both causal modes, multiple heads, and tail tiles
actual,lse=flash_forward_triton(q,k,v,return_lse=True)
expected,ref_lse=attention_torch(q,k,v,return_lse=True)
torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
torch.testing.assert_close(lse,ref_lse,rtol=1e-5,atol=1e-5)
print("PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
print("Max error:",(actual-expected).abs().max().item())
for b,h,length,d in ((0,2,3,7),(1,1,1,1),(2,3,17,7),(1,2,33,31),(1,2,65,64),(1,1,5,128)):
    queries=torch.randn(b,h,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    for causal in (False,True):
        out,saved=flash_forward_triton(queries,keys,values,causal,True)
        ref,normalizer=attention_torch(queries,keys,values,causal,True)
        torch.testing.assert_close(out,ref,rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(saved,normalizer,rtol=1e-5,atol=1e-5)
print("All batch/head, causal, empty-batch, and padding checks passed.")


# 7. Compare eager, function, and Triton multi-head times
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")

print("B,H,T,D       causal  eager us  function us  Triton us  speedup  max error")
for b,h,length,d in ((1,1,32,32),(1,4,128,64),(2,4,256,64)):
    queries=torch.randn(b,h,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    mask=torch.ones(length,length,device=DEVICE,dtype=torch.bool).tril()
    for causal in (False,True):
        # Direct eager operators, with a reusable mask for causal mode.
        def eager_call():
            s=(queries@keys.transpose(-1,-2))*(d**-0.5)
            if causal:
                s=s.masked_fill(~mask,-float("inf"))
            return torch.softmax(s,dim=-1)@values
        expected=attention_torch(queries,keys,values,causal)
        actual=flash_forward_triton(queries,keys,values,causal)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        error=(actual-expected).abs().max().item()
        e=bench_ms(eager_call)
        p=bench_ms(lambda:attention_torch(queries,keys,values,causal))
        t=bench_ms(lambda:flash_forward_triton(queries,keys,values,causal))
        print(f"{b},{h},{length},{d} {str(causal):>7} {e*1000:9.3f} {p*1000:12.3f} {t*1000:10.3f} {p/t:7.2f}x {error:.3e}")


# 8. Check head independence and concatenate head features
original=flash_forward_triton(q,k,v)
changed_values=v.clone()
changed_values[:,0]+=5.0
changed=flash_forward_triton(q,k,changed_values)
torch.testing.assert_close(changed[:,1:],original[:,1:],rtol=0,atol=0)
torch.testing.assert_close(changed[:,0],original[:,0]+5.0,rtol=1e-5,atol=1e-5)
b,h,length,d=original.shape
merged=original.transpose(1,2).contiguous().reshape(b,length,h*d)
print("Independent heads passed; concatenated shape:",tuple(merged.shape))
print("Concatenated output:\n",merged.cpu())
```
