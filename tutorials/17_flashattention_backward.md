# 17. FlashAttention backward: recompute probabilities and accumulate gradients

Previous: [multi-head FlashAttention](16_multi_head_flashattention.md). Next: [quantization / dequantization](18_quantization_dequantization.md). Related: [online softmax](13_online_softmax.md) and [FlashAttention forward](14_flashattention_forward.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are contiguous FP32 tensors of finite values on one NVIDIA CUDA GPU, with moderate magnitudes whose products and sums remain representable in FP32. Fixed tiles and IEEE dot products make this a teaching implementation of the FlashAttention algorithm; optimized Tensor Core implementations have different performance characteristics. No dropout is used. This guide adds first-order gradients for matching `(B,H,T,D)` Q/K/V tensors and supports both unmasked and square causal self-attention. It includes all forward code needed by backward and a custom PyTorch autograd function. T=1–4096 and D=1–128; B*H is at most 65535.

## Concepts: differentiate attention without saving its full probabilities

Let $a=1/\sqrt D$, $S=aQK^T$, $P=\operatorname{softmax}(S)$ along keys, and $O=PV$. An upstream gradient $G=\partial\mathcal L/\partial O$ has the same shape as O. The equations below describe one batch/head pair; the computation is independent across all other pairs. With causal masking, forbidden scores are negative infinity and their probabilities and score gradients are zero.

### 1. Start with the final value product

Differentiating $O=PV$ gives

$$
dV=P^TG,\qquad dP=GV^T.
$$

The shape of dV is `(T,D)` and the conceptual shape of dP is `(T,T)`. We will compute dP only as tiles, consume those tiles immediately, and avoid storing the full matrix.

### 2. Differentiate a softmax row

For one row, $\partial p_j/\partial s_k=p_j(\mathbf1_{j=k}-p_k)$. Applying this Jacobian to dP yields

$$
dS_{i,j}=P_{i,j}\left(dP_{i,j}-\sum_k P_{i,k}dP_{i,k}\right).
$$

The subtraction couples every allowed key in the row. Omitting it would treat the denominator as constant and give incorrect gradients. Causal keys have $P_{i,j}=0$, so their dS is zero after the same formula.

### 3. Replace a quadratic row reduction with a saved-output identity

Define the row scalar $\delta_i=\sum_j P_{i,j}dP_{i,j}$. Substitute $dP_{i,j}=\sum_f G_{i,f}V_{j,f}$ and exchange the two sums:

$$
\delta_i=\sum_f G_{i,f}\left(\sum_j P_{i,j}V_{j,f}\right)
=\boxed{\sum_f G_{i,f}O_{i,f}}.
$$

O is already saved from forward. A short feature reduction therefore computes delta before the tiled backward pass, without reconstructing every key just to find a row reduction. The resulting tile formula is

$$
\boxed{dS_{i,j}=P_{i,j}(G_i\cdot V_j-\delta_i)}.
$$

This identity is exact in real arithmetic; finite-precision saved outputs and recomputed probabilities introduce small numerical differences. The checks compare gradients using explicit tolerances.

### 4. Differentiate the scaled score product

Since $S=aQK^T$,

$$
\boxed{dQ=a\,dS\,K},\qquad
\boxed{dK=a\,dS^TQ},\qquad
\boxed{dV=P^TG}.
$$

The scale multiplies dQ and dK because they differentiate S. It does not multiply dV, which differentiates O's value product. In this implementation dS denotes the derivative with respect to the **scaled** scores, and the scale is applied when storing dQ/dK.

```text
saved O + upstream G -> delta = row_sum(O * G)
                              |
saved Q/K -> score tile       |
saved LSE -> P = exp(S - LSE)  |
V + G     -> dP = G @ Vᵀ      |
              +---------------+
              |
       dS = P * (dP - delta)
              |
       +------+---------+
       |                |
  dQ += a*dS@K     dK += a*dSᵀ@Q
                   dV += Pᵀ@G
```

### 5. Online forward statistics make recomputation possible

Forward maintained a maximum $m_i$, shifted denominator $\ell_i$, and vector numerator $u_i$. Every new block rescales **both** $\ell_i$ and $u_i$ by $e^{m_i-m_i'}$. The final output and saved statistic are

$$
O_i=u_i/\ell_i,\qquad L_i=m_i+\log\ell_i.
$$

Backward reconstructs each allowed probability tile with $P_{i,j}=e^{S_{i,j}-L_i}$. A causal mask must be applied again before exponentiation. LSE stores natural logarithms; this code uses `tl.exp`, so the units match. Invalid padded rows load a harmless LSE of zero, and their score mask forces P to zero.

A stable forward normalization is essential: an incorrect numerator correction corrupts saved O and therefore delta, while an incorrect denominator correction corrupts LSE and therefore every reconstructed P. This connects the online recurrence directly to backward correctness.

The saved forward data are Q/K/V, O, and one LSE value per query. No full S, P, dP, or dS tensor is allocated by the Triton implementation. Retaining Q/K/V does not duplicate them; the autograd context keeps references. Backward uses one additional row-sized delta buffer and the three required gradient tensors. Auxiliary storage grows linearly in T for fixed B/H/D.

### 6. Choose output ownership to avoid atomics

This teaching implementation uses three backward launches:

1. **Delta:** one program reduces one row's feature products O*G.
2. **dQ:** one program owns a query tile and scans keys, accumulating that tile's dQ.
3. **dK/dV:** one program owns a key tile and scans queries, accumulating that tile's dK and dV.

Each gradient tile has one writer, so no atomic additions or cross-program reduction buffers are needed. The dQ and dK/dV passes recompute overlapping score tiles independently; this spends additional arithmetic to keep ownership clear and storage bounded. In causal mode the dQ scan skips fully future key tiles, while the simple dK/dV pass scans all query tiles and masks forbidden pairs.

This is a complete learning implementation rather than the schedule used by every optimized FlashAttention kernel. It retains quadratic arithmetic and adds recomputation; lower intermediate storage does not guarantee faster execution for every shape. The [FlashAttention paper](https://arxiv.org/abs/2205.14135) explains the memory/recomputation tradeoff, and the [official Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) demonstrates a more tuned backward implementation.

### 7. Connect explicit kernels to PyTorch autograd

A direct Triton launch writes a tensor but does not tell PyTorch how to differentiate it. `torch.autograd.Function` supplies that rule: forward saves Q/K/V/O/LSE and the causal flag, and backward returns dQ/dK/dV plus `None` for the non-tensor flag. An upstream gradient can be noncontiguous, so the autograd wrapper makes it contiguous before launching.

The custom operation supports first-order gradients only; `once_differentiable` explicitly prevents treating its raw backward launches as a differentiable second-order graph. There is no dropout, arbitrary attention mask, unequal sequence lengths, or gradient for the fixed scaling factor in this notebook. For the tests, the scalar loss $\mathcal L=\sum_{i,f}O_{i,f}G_{i,f}$ supplies exactly the chosen upstream G.

## 1. Imports, tiny inputs, and an upstream gradient

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
from torch.autograd.function import once_differentiable

q=torch.tensor([[[[1.,0.],[0.,1.]]]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[[[10.,0.],[0.,20.]]]],device=DEVICE,dtype=DTYPE)
do=torch.tensor([[[[0.5,-1.],[1.,0.25]]]],device=DEVICE,dtype=DTYPE)
```

`do` is the gradient with respect to the attention output. Keeping it explicit makes the three input gradients reproducible.

## 2. PyTorch eager: calculate the gradient equations directly

Begin with the unmasked problem and expose every derivative.

```python
scale=q.shape[-1]**-0.5
scores=(q@k.transpose(-1,-2))*scale
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
dp=do@v.transpose(-1,-2)
delta=(o_eager*do).sum(dim=-1,keepdim=True)
ds=probabilities*(dp-delta)
dq_eager=(ds@k)*scale
dk_eager=(ds.transpose(-1,-2)@q)*scale
dv_eager=probabilities.transpose(-1,-2)@do
print("Output:\n",o_eager.cpu())
print("Delta:\n",delta.cpu())
for name,gradient in zip(("dQ","dK","dV"),(dq_eager,dk_eager,dv_eager)):
    print(name,gradient.cpu().tolist())
```

These are the formulas derived above, expressed as dense eager operations. For causal attention, masking the scores before softmax also makes future dS entries zero.

## 3. PyTorch function: get an independent autograd reference

Compare the manual equations with gradients produced by PyTorch.

```python
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


def pytorch_forward_backward(q,k,v,do,causal=False):
    with torch.enable_grad():
        leaves=[t.detach().requires_grad_(True) for t in (q,k,v)]
        out=attention_torch(*leaves,causal=causal)
        gradients=torch.autograd.grad(out,leaves,grad_outputs=do)
    return out,gradients


ref_out,ref_gradients=pytorch_forward_backward(q,k,v,do)
torch.testing.assert_close(ref_out,o_eager,rtol=0,atol=0)
for manual,reference in zip((dq_eager,dk_eager,dv_eager),ref_gradients):
    torch.testing.assert_close(manual,reference,rtol=1e-5,atol=1e-5)
print("Manual gradient equations agree with PyTorch autograd.")
```

The function uses fresh leaf views and `torch.autograd.grad`, so repeated benchmark calls do not accumulate into `.grad`. `detach()` shares input storage; it does not copy the GPU inputs.

## 4. Triton: forward statistics and three backward kernels

Keep the saved forward kernel in the notebook so the gradient code is self-contained.

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


@triton.jit
def flash_delta_kernel(O, DO, Delta, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    features = tl.arange(0,BLOCK_D)
    output = tl.load(O + row*D + features,mask=features<D,other=0.0)
    upstream = tl.load(DO + row*D + features,mask=features<D,other=0.0)
    tl.store(Delta + row,tl.sum(output*upstream,axis=0))


@triton.jit
def flash_dq_kernel(
    Q, K, V, DO, LSE, Delta, DQ, Scale,
    T: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # This program owns a query tile's dQ; no other program writes it.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile*BLOCK_M + tl.arange(0,BLOCK_M)
    features = tl.arange(0,BLOCK_D)
    local_keys = tl.arange(0,BLOCK_N)
    q = tl.load(Q+bh*T*D+rows[:,None]*D+features[None,:],
                mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
    do = tl.load(DO+bh*T*D+rows[:,None]*D+features[None,:],
                 mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
    lse = tl.load(LSE+bh*T+rows,mask=rows<T,other=0.0)
    delta = tl.load(Delta+bh*T+rows,mask=rows<T,other=0.0)
    dq = tl.zeros((BLOCK_M,BLOCK_D),tl.float32)
    end = T
    if CAUSAL:
        end = tl.minimum(T,(query_tile+1)*BLOCK_M)
    for start in range(0,end,BLOCK_N):
        keys = start+local_keys
        kt = tl.load(K+bh*T*D+keys[None,:]*D+features[:,None],
                     mask=(keys[None,:]<T)&(features[:,None]<D),other=0.0)
        vt = tl.load(V+bh*T*D+keys[None,:]*D+features[:,None],
                     mask=(keys[None,:]<T)&(features[:,None]<D),other=0.0)
        scores = tl.dot(q,kt,input_precision="ieee")*Scale
        allowed = (rows[:,None]<T)&(keys[None,:]<T)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        p = tl.exp(scores-lse[:,None])
        dp = tl.dot(do,vt,input_precision="ieee")
        ds = p*(dp-delta[:,None])
        dq = tl.dot(ds,tl.trans(kt),dq,input_precision="ieee")
    tl.store(DQ+bh*T*D+rows[:,None]*D+features[None,:],dq*Scale,
             mask=(rows[:,None]<T)&(features[None,:]<D))


@triton.jit
def flash_dkdv_kernel(
    Q, K, V, DO, LSE, Delta, DK, DV, Scale,
    T: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # This program owns one key tile's dK and dV across all queries.
    bh = tl.program_id(1)
    keys = tl.program_id(0)*BLOCK_N + tl.arange(0,BLOCK_N)
    features = tl.arange(0,BLOCK_D)
    local_rows = tl.arange(0,BLOCK_M)
    k = tl.load(K+bh*T*D+keys[:,None]*D+features[None,:],
                mask=(keys[:,None]<T)&(features[None,:]<D),other=0.0)
    v = tl.load(V+bh*T*D+keys[:,None]*D+features[None,:],
                mask=(keys[:,None]<T)&(features[None,:]<D),other=0.0)
    dk = tl.zeros((BLOCK_N,BLOCK_D),tl.float32)
    dv = tl.zeros((BLOCK_N,BLOCK_D),tl.float32)
    for start in range(0,T,BLOCK_M):
        rows = start+local_rows
        q = tl.load(Q+bh*T*D+rows[:,None]*D+features[None,:],
                    mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
        do = tl.load(DO+bh*T*D+rows[:,None]*D+features[None,:],
                     mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
        lse = tl.load(LSE+bh*T+rows,mask=rows<T,other=0.0)
        delta = tl.load(Delta+bh*T+rows,mask=rows<T,other=0.0)
        scores = tl.dot(q,tl.trans(k),input_precision="ieee")*Scale
        allowed = (rows[:,None]<T)&(keys[None,:]<T)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        p = tl.exp(scores-lse[:,None])
        dp = tl.dot(do,tl.trans(v),input_precision="ieee")
        ds = p*(dp-delta[:,None])
        dk = tl.dot(tl.trans(ds),q,dk,input_precision="ieee")
        dv = tl.dot(tl.trans(p),do,dv,input_precision="ieee")
    tl.store(DK+bh*T*D+keys[:,None]*D+features[None,:],dk*Scale,
             mask=(keys[:,None]<T)&(features[None,:]<D))
    tl.store(DV+bh*T*D+keys[:,None]*D+features[None,:],dv,
             mask=(keys[:,None]<T)&(features[None,:]<D))
```

Read the kernels in order:

1. Forward streams K/V, rescales both online accumulators, and stores O/LSE.
2. Delta reduces the feature axis of O*G for every real row.
3. dQ loads a query tile and its G/LSE/delta, reconstructs P, forms dS, and accumulates dS@K.
4. dK/dV loads a key tile, loops over query tiles, and accumulates dSᵀ@Q and Pᵀ@G.
5. dQ/dK apply the score scale at their final stores; dV has no extra scale.

Every load and store masks sequence and feature padding. The causal mask appears in both backward score-reconstruction passes.

## 5. Python wrappers and a custom autograd operation

The raw backward consumes outputs and statistics from the matching forward.

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


def flash_backward_triton(q,k,v,out,lse,do,causal=False):
    # out/lse must come from the matching forward call and causal mode.
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v,out,lse,do)):
        raise ValueError("Saved tensors and upstream gradient must share the CUDA device.")
    if q.ndim!=4 or not all(t.shape==q.shape for t in (k,v,out,do)):
        raise ValueError("Q/K/V/output/upstream must have matching rank-four shapes.")
    if lse.shape!=q.shape[:3]:
        raise ValueError("LSE must have shape (batch,head,length).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v,out,lse,do)):
        raise ValueError("Saved tensors and upstream must be contiguous FP32 tensors.")
    batches,heads,length,d=q.shape
    if not 1<=length<=4096 or not 1<=d<=128 or batches*heads>65535:
        raise ValueError("Use T=1–4096, D=1–128, and at most 65535 batch/head pairs.")
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    delta=torch.empty_like(lse)
    if batches*heads:
        block_d=max(32,triton.next_power_of_2(d))
        grid=(triton.cdiv(length,32),batches*heads)
        with torch.cuda.device(q.device):
            flash_delta_kernel[(batches*heads*length,)](out,do,delta,D=d,BLOCK_D=block_d,num_warps=4)
            flash_dq_kernel[grid](q,k,v,do,lse,delta,dq,d**-0.5,
                                 T=length,D=d,CAUSAL=causal,BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4)
            flash_dkdv_kernel[grid](q,k,v,do,lse,delta,dk,dv,d**-0.5,
                                   T=length,D=d,CAUSAL=causal,BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4)
    return dq,dk,dv


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,causal):
        out,lse=flash_forward_triton(q,k,v,causal,return_lse=True)
        ctx.save_for_backward(q,k,v,out,lse)
        ctx.causal=causal
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,do):
        q,k,v,out,lse=ctx.saved_tensors
        dq,dk,dv=flash_backward_triton(q,k,v,out,lse,do.contiguous(),ctx.causal)
        return dq,dk,dv,None


def flash_attention_autograd(q,k,v,causal=False):
    return FlashAttentionFunction.apply(q,k,v,causal)
```

The raw API assumes `out` and `lse` came from these exact Q/K/V inputs and the same causal mode. Shape validation cannot check that provenance. The autograd context automatically supplies the matching saved tensors. It returns `None` for the causal flag because that argument has no gradient.

## 6. Compare gradient values, tail tiles, and autograd integration

Check all three gradients in unmasked and causal modes.

```python
out,lse=flash_forward_triton(q,k,v,return_lse=True)
actual_gradients=flash_backward_triton(q,k,v,out,lse,do)
expected_out,expected_gradients=pytorch_forward_backward(q,k,v,do)
torch.testing.assert_close(out,expected_out,rtol=1e-4,atol=1e-4)
for name,actual,expected in zip(("dQ","dK","dV"),actual_gradients,expected_gradients):
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    print(name,"PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
    print(name,"max error:",(actual-expected).abs().max().item())

for length,d in ((1,1),(17,7),(33,31),(65,64),(5,128)):
    queries=torch.randn(1,2,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    upstream=torch.randn_like(queries)
    for causal in (False,True):
        result,saved=flash_forward_triton(queries,keys,values,causal,True)
        gradients=flash_backward_triton(queries,keys,values,result,saved,upstream,causal)
        reference,ref_grads=pytorch_forward_backward(queries,keys,values,upstream,causal)
        torch.testing.assert_close(result,reference,rtol=1e-4,atol=1e-4)
        for actual,expected in zip(gradients,ref_grads):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        leaves=[t.detach().requires_grad_(True) for t in (queries,keys,values)]
        integrated=flash_attention_autograd(*leaves,causal=causal)
        loss=(integrated*upstream).sum()
        integrated_grads=torch.autograd.grad(loss,leaves)
        for actual,expected in zip(integrated_grads,ref_grads):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
print("All forward, dQ/dK/dV, causal, padding, and autograd checks passed.")
```

D=1 and non-power-of-two dimensions check scaling and padding. T=33 and T=65 cross tile boundaries. Using two heads checks that backward offsets remain independent.

## 7. Compare complete forward-plus-backward times

Include forward statistics, delta, and all three input gradients.

```python
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")


def triton_forward_backward(q,k,v,do,causal=False):
    out,lse=flash_forward_triton(q,k,v,causal,True)
    return out,flash_backward_triton(q,k,v,out,lse,do,causal)


def integrated_forward_backward(q,k,v,do,causal=False):
    with torch.enable_grad():
        leaves=[t.detach().requires_grad_(True) for t in (q,k,v)]
        out=flash_attention_autograd(*leaves,causal=causal)
        gradients=torch.autograd.grad(out,leaves,grad_outputs=do)
    return out,gradients


print("T,D     causal  PyTorch us  raw Triton us  autograd Triton us  speedup  max grad error")
for length,d in ((32,32),(128,64),(256,64)):
    queries=torch.randn(1,2,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    upstream=torch.randn_like(queries)
    for causal in (False,True):
        expected,ref_grads=pytorch_forward_backward(queries,keys,values,upstream,causal)
        actual,grads=triton_forward_backward(queries,keys,values,upstream,causal)
        integrated,integrated_grads=integrated_forward_backward(queries,keys,values,upstream,causal)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(integrated,expected,rtol=1e-4,atol=1e-4)
        for tested in (grads,integrated_grads):
            for gradient,reference in zip(tested,ref_grads):
                torch.testing.assert_close(gradient,reference,rtol=1e-4,atol=1e-4)
        error=max((a-b).abs().max().item() for a,b in zip(grads,ref_grads))
        p=bench_ms(lambda:pytorch_forward_backward(queries,keys,values,upstream,causal))
        t=bench_ms(lambda:triton_forward_backward(queries,keys,values,upstream,causal))
        a=bench_ms(lambda:integrated_forward_backward(queries,keys,values,upstream,causal))
        print(f"{length},{d} {str(causal):>7} {p*1000:11.3f} {t*1000:14.3f} {a*1000:19.3f} {p/t:7.2f}x {error:.3e}")
```

[`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. Input creation, first compilation, correctness checks, CPU copies, and printing are outside timing. Each callable allocates its outputs. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock measurements. Speedup is PyTorch function time divided by Triton time. Run the code to get actual measurements; no speedup is assumed. Here speedup uses the raw Triton forward-plus-backward time. The additional integrated column measures the custom operation called through PyTorch autograd. The PyTorch baseline explicitly materializes dense attention and builds its autograd graph each call; it does not invoke an optimized SDPA backend.

## 8. Measure backward alone with forward prepared beforehand

Retain one PyTorch forward graph and one Triton forward result outside timing.

```python
queries=torch.randn(1,2,128,64,device=DEVICE,dtype=DTYPE)
keys=torch.randn_like(queries)
values=torch.randn_like(queries)
upstream=torch.randn_like(queries)
causal=True
leaves=[t.detach().requires_grad_(True) for t in (queries,keys,values)]
reference_output=attention_torch(*leaves,causal=causal)
output,saved=flash_forward_triton(queries,keys,values,causal,True)


def pytorch_backward_only():
    # Keep the forward graph reusable; every call returns new gradients.
    return torch.autograd.grad(reference_output,leaves,grad_outputs=upstream,retain_graph=True)


def triton_backward_only():
    return flash_backward_triton(queries,keys,values,output,saved,upstream,causal)


reference=pytorch_backward_only()
actual=triton_backward_only()
for a,b in zip(actual,reference):
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
p=bench_ms(pytorch_backward_only)
t=bench_ms(triton_backward_only)
print(f"Backward only: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, speedup {p/t:.2f}x")
```

Triton timing includes the delta launch and both gradient launches, output allocation, and probability recomputation. PyTorch timing reuses its saved dense forward intermediates; Triton reuses Q/K/V/O/LSE. These are the respective backward algorithms, with their own saved-state requirements. The preceding table separately reports complete forward-plus-backward cost.

## Try next

Continue with [quantization / dequantization](18_quantization_dequantization.md) and [quantized GEMM / FP8](19_quantized_gemm_fp8.md). Try tuning tiles and lower-precision dot products only after preserving the output and gradient checks. For an established optimized API, compare against [PyTorch scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), documenting its selected backend and dtype separately from the explicit dense baseline used here.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code, including imports, PyTorch references, Triton kernels, wrappers, checks, and benchmarks. Comments mark each step.

```python
# 1. Imports, tiny inputs, and an upstream gradient
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
from torch.autograd.function import once_differentiable

q=torch.tensor([[[[1.,0.],[0.,1.]]]],device=DEVICE,dtype=DTYPE)
k=q.clone()
v=torch.tensor([[[[10.,0.],[0.,20.]]]],device=DEVICE,dtype=DTYPE)
do=torch.tensor([[[[0.5,-1.],[1.,0.25]]]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: calculate the gradient equations directly
scale=q.shape[-1]**-0.5
scores=(q@k.transpose(-1,-2))*scale
probabilities=torch.softmax(scores,dim=-1)
o_eager=probabilities@v
dp=do@v.transpose(-1,-2)
delta=(o_eager*do).sum(dim=-1,keepdim=True)
ds=probabilities*(dp-delta)
dq_eager=(ds@k)*scale
dk_eager=(ds.transpose(-1,-2)@q)*scale
dv_eager=probabilities.transpose(-1,-2)@do
print("Output:\n",o_eager.cpu())
print("Delta:\n",delta.cpu())
for name,gradient in zip(("dQ","dK","dV"),(dq_eager,dk_eager,dv_eager)):
    print(name,gradient.cpu().tolist())


# 3. PyTorch function: get an independent autograd reference
def attention_torch(q,k,v,causal=False,return_lse=False):
    scores=(q@k.transpose(-1,-2))*(q.shape[-1]**-0.5)
    if causal:
        allowed=torch.ones(q.shape[-2],k.shape[-2],device=q.device,dtype=torch.bool).tril()
        scores=scores.masked_fill(~allowed,-float("inf"))
    out=torch.softmax(scores,dim=-1)@v
    return (out,torch.logsumexp(scores,dim=-1)) if return_lse else out


def pytorch_forward_backward(q,k,v,do,causal=False):
    with torch.enable_grad():
        leaves=[t.detach().requires_grad_(True) for t in (q,k,v)]
        out=attention_torch(*leaves,causal=causal)
        gradients=torch.autograd.grad(out,leaves,grad_outputs=do)
    return out,gradients


ref_out,ref_gradients=pytorch_forward_backward(q,k,v,do)
torch.testing.assert_close(ref_out,o_eager,rtol=0,atol=0)
for manual,reference in zip((dq_eager,dk_eager,dv_eager),ref_gradients):
    torch.testing.assert_close(manual,reference,rtol=1e-5,atol=1e-5)
print("Manual gradient equations agree with PyTorch autograd.")


# 4. Triton: forward statistics and three backward kernels
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


@triton.jit
def flash_delta_kernel(O, DO, Delta, D: tl.constexpr, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    features = tl.arange(0,BLOCK_D)
    output = tl.load(O + row*D + features,mask=features<D,other=0.0)
    upstream = tl.load(DO + row*D + features,mask=features<D,other=0.0)
    tl.store(Delta + row,tl.sum(output*upstream,axis=0))


@triton.jit
def flash_dq_kernel(
    Q, K, V, DO, LSE, Delta, DQ, Scale,
    T: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # This program owns a query tile's dQ; no other program writes it.
    query_tile = tl.program_id(0)
    bh = tl.program_id(1)
    rows = query_tile*BLOCK_M + tl.arange(0,BLOCK_M)
    features = tl.arange(0,BLOCK_D)
    local_keys = tl.arange(0,BLOCK_N)
    q = tl.load(Q+bh*T*D+rows[:,None]*D+features[None,:],
                mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
    do = tl.load(DO+bh*T*D+rows[:,None]*D+features[None,:],
                 mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
    lse = tl.load(LSE+bh*T+rows,mask=rows<T,other=0.0)
    delta = tl.load(Delta+bh*T+rows,mask=rows<T,other=0.0)
    dq = tl.zeros((BLOCK_M,BLOCK_D),tl.float32)
    end = T
    if CAUSAL:
        end = tl.minimum(T,(query_tile+1)*BLOCK_M)
    for start in range(0,end,BLOCK_N):
        keys = start+local_keys
        kt = tl.load(K+bh*T*D+keys[None,:]*D+features[:,None],
                     mask=(keys[None,:]<T)&(features[:,None]<D),other=0.0)
        vt = tl.load(V+bh*T*D+keys[None,:]*D+features[:,None],
                     mask=(keys[None,:]<T)&(features[:,None]<D),other=0.0)
        scores = tl.dot(q,kt,input_precision="ieee")*Scale
        allowed = (rows[:,None]<T)&(keys[None,:]<T)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        p = tl.exp(scores-lse[:,None])
        dp = tl.dot(do,vt,input_precision="ieee")
        ds = p*(dp-delta[:,None])
        dq = tl.dot(ds,tl.trans(kt),dq,input_precision="ieee")
    tl.store(DQ+bh*T*D+rows[:,None]*D+features[None,:],dq*Scale,
             mask=(rows[:,None]<T)&(features[None,:]<D))


@triton.jit
def flash_dkdv_kernel(
    Q, K, V, DO, LSE, Delta, DK, DV, Scale,
    T: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # This program owns one key tile's dK and dV across all queries.
    bh = tl.program_id(1)
    keys = tl.program_id(0)*BLOCK_N + tl.arange(0,BLOCK_N)
    features = tl.arange(0,BLOCK_D)
    local_rows = tl.arange(0,BLOCK_M)
    k = tl.load(K+bh*T*D+keys[:,None]*D+features[None,:],
                mask=(keys[:,None]<T)&(features[None,:]<D),other=0.0)
    v = tl.load(V+bh*T*D+keys[:,None]*D+features[None,:],
                mask=(keys[:,None]<T)&(features[None,:]<D),other=0.0)
    dk = tl.zeros((BLOCK_N,BLOCK_D),tl.float32)
    dv = tl.zeros((BLOCK_N,BLOCK_D),tl.float32)
    for start in range(0,T,BLOCK_M):
        rows = start+local_rows
        q = tl.load(Q+bh*T*D+rows[:,None]*D+features[None,:],
                    mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
        do = tl.load(DO+bh*T*D+rows[:,None]*D+features[None,:],
                     mask=(rows[:,None]<T)&(features[None,:]<D),other=0.0)
        lse = tl.load(LSE+bh*T+rows,mask=rows<T,other=0.0)
        delta = tl.load(Delta+bh*T+rows,mask=rows<T,other=0.0)
        scores = tl.dot(q,tl.trans(k),input_precision="ieee")*Scale
        allowed = (rows[:,None]<T)&(keys[None,:]<T)
        if CAUSAL:
            allowed = allowed & (keys[None,:] <= rows[:,None])
        scores = tl.where(allowed,scores,-float("inf"))
        p = tl.exp(scores-lse[:,None])
        dp = tl.dot(do,tl.trans(v),input_precision="ieee")
        ds = p*(dp-delta[:,None])
        dk = tl.dot(tl.trans(ds),q,dk,input_precision="ieee")
        dv = tl.dot(tl.trans(p),do,dv,input_precision="ieee")
    tl.store(DK+bh*T*D+keys[:,None]*D+features[None,:],dk*Scale,
             mask=(keys[:,None]<T)&(features[None,:]<D))
    tl.store(DV+bh*T*D+keys[:,None]*D+features[None,:],dv,
             mask=(keys[:,None]<T)&(features[None,:]<D))


# 5. Python wrappers and a custom autograd operation
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


def flash_backward_triton(q,k,v,out,lse,do,causal=False):
    # out/lse must come from the matching forward call and causal mode.
    if not all(t.is_cuda and t.device==q.device for t in (q,k,v,out,lse,do)):
        raise ValueError("Saved tensors and upstream gradient must share the CUDA device.")
    if q.ndim!=4 or not all(t.shape==q.shape for t in (k,v,out,do)):
        raise ValueError("Q/K/V/output/upstream must have matching rank-four shapes.")
    if lse.shape!=q.shape[:3]:
        raise ValueError("LSE must have shape (batch,head,length).")
    if not all(t.dtype==torch.float32 and t.is_contiguous() for t in (q,k,v,out,lse,do)):
        raise ValueError("Saved tensors and upstream must be contiguous FP32 tensors.")
    batches,heads,length,d=q.shape
    if not 1<=length<=4096 or not 1<=d<=128 or batches*heads>65535:
        raise ValueError("Use T=1–4096, D=1–128, and at most 65535 batch/head pairs.")
    dq,dk,dv=torch.empty_like(q),torch.empty_like(k),torch.empty_like(v)
    delta=torch.empty_like(lse)
    if batches*heads:
        block_d=max(32,triton.next_power_of_2(d))
        grid=(triton.cdiv(length,32),batches*heads)
        with torch.cuda.device(q.device):
            flash_delta_kernel[(batches*heads*length,)](out,do,delta,D=d,BLOCK_D=block_d,num_warps=4)
            flash_dq_kernel[grid](q,k,v,do,lse,delta,dq,d**-0.5,
                                 T=length,D=d,CAUSAL=causal,BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4)
            flash_dkdv_kernel[grid](q,k,v,do,lse,delta,dk,dv,d**-0.5,
                                   T=length,D=d,CAUSAL=causal,BLOCK_M=32,BLOCK_N=32,BLOCK_D=block_d,num_warps=4)
    return dq,dk,dv


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,causal):
        out,lse=flash_forward_triton(q,k,v,causal,return_lse=True)
        ctx.save_for_backward(q,k,v,out,lse)
        ctx.causal=causal
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,do):
        q,k,v,out,lse=ctx.saved_tensors
        dq,dk,dv=flash_backward_triton(q,k,v,out,lse,do.contiguous(),ctx.causal)
        return dq,dk,dv,None


def flash_attention_autograd(q,k,v,causal=False):
    return FlashAttentionFunction.apply(q,k,v,causal)


# 6. Compare gradient values, tail tiles, and autograd integration
out,lse=flash_forward_triton(q,k,v,return_lse=True)
actual_gradients=flash_backward_triton(q,k,v,out,lse,do)
expected_out,expected_gradients=pytorch_forward_backward(q,k,v,do)
torch.testing.assert_close(out,expected_out,rtol=1e-4,atol=1e-4)
for name,actual,expected in zip(("dQ","dK","dV"),actual_gradients,expected_gradients):
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
    print(name,"PyTorch:",expected.cpu().tolist(),"Triton:",actual.cpu().tolist())
    print(name,"max error:",(actual-expected).abs().max().item())

for length,d in ((1,1),(17,7),(33,31),(65,64),(5,128)):
    queries=torch.randn(1,2,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    upstream=torch.randn_like(queries)
    for causal in (False,True):
        result,saved=flash_forward_triton(queries,keys,values,causal,True)
        gradients=flash_backward_triton(queries,keys,values,result,saved,upstream,causal)
        reference,ref_grads=pytorch_forward_backward(queries,keys,values,upstream,causal)
        torch.testing.assert_close(result,reference,rtol=1e-4,atol=1e-4)
        for actual,expected in zip(gradients,ref_grads):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        leaves=[t.detach().requires_grad_(True) for t in (queries,keys,values)]
        integrated=flash_attention_autograd(*leaves,causal=causal)
        loss=(integrated*upstream).sum()
        integrated_grads=torch.autograd.grad(loss,leaves)
        for actual,expected in zip(integrated_grads,ref_grads):
            torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
print("All forward, dQ/dK/dV, causal, padding, and autograd checks passed.")


# 7. Compare complete forward-plus-backward times
def bench_ms(fn):
    # Compile and warm up before timing steady-state execution.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn,warmup=25,rep=100,return_mode="median")


def triton_forward_backward(q,k,v,do,causal=False):
    out,lse=flash_forward_triton(q,k,v,causal,True)
    return out,flash_backward_triton(q,k,v,out,lse,do,causal)


def integrated_forward_backward(q,k,v,do,causal=False):
    with torch.enable_grad():
        leaves=[t.detach().requires_grad_(True) for t in (q,k,v)]
        out=flash_attention_autograd(*leaves,causal=causal)
        gradients=torch.autograd.grad(out,leaves,grad_outputs=do)
    return out,gradients


print("T,D     causal  PyTorch us  raw Triton us  autograd Triton us  speedup  max grad error")
for length,d in ((32,32),(128,64),(256,64)):
    queries=torch.randn(1,2,length,d,device=DEVICE,dtype=DTYPE)
    keys=torch.randn_like(queries)
    values=torch.randn_like(queries)
    upstream=torch.randn_like(queries)
    for causal in (False,True):
        expected,ref_grads=pytorch_forward_backward(queries,keys,values,upstream,causal)
        actual,grads=triton_forward_backward(queries,keys,values,upstream,causal)
        integrated,integrated_grads=integrated_forward_backward(queries,keys,values,upstream,causal)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(integrated,expected,rtol=1e-4,atol=1e-4)
        for tested in (grads,integrated_grads):
            for gradient,reference in zip(tested,ref_grads):
                torch.testing.assert_close(gradient,reference,rtol=1e-4,atol=1e-4)
        error=max((a-b).abs().max().item() for a,b in zip(grads,ref_grads))
        p=bench_ms(lambda:pytorch_forward_backward(queries,keys,values,upstream,causal))
        t=bench_ms(lambda:triton_forward_backward(queries,keys,values,upstream,causal))
        a=bench_ms(lambda:integrated_forward_backward(queries,keys,values,upstream,causal))
        print(f"{length},{d} {str(causal):>7} {p*1000:11.3f} {t*1000:14.3f} {a*1000:19.3f} {p/t:7.2f}x {error:.3e}")


# 8. Measure backward alone with forward prepared beforehand
queries=torch.randn(1,2,128,64,device=DEVICE,dtype=DTYPE)
keys=torch.randn_like(queries)
values=torch.randn_like(queries)
upstream=torch.randn_like(queries)
causal=True
leaves=[t.detach().requires_grad_(True) for t in (queries,keys,values)]
reference_output=attention_torch(*leaves,causal=causal)
output,saved=flash_forward_triton(queries,keys,values,causal,True)


def pytorch_backward_only():
    # Keep the forward graph reusable; every call returns new gradients.
    return torch.autograd.grad(reference_output,leaves,grad_outputs=upstream,retain_graph=True)


def triton_backward_only():
    return flash_backward_triton(queries,keys,values,output,saved,upstream,causal)


reference=pytorch_backward_only()
actual=triton_backward_only()
for a,b in zip(actual,reference):
    torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-4)
p=bench_ms(pytorch_backward_only)
t=bench_ms(triton_backward_only)
print(f"Backward only: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, speedup {p/t:.2f}x")
```
