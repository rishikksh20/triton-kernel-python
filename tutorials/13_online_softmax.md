# 13. Online softmax: stream the maximum and denominator together

Previous: [attention without Flash](12_attention_without_flash.md). Next: [FlashAttention forward](14_flashattention_forward.md). Related: [row-wise softmax](06_softmax.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. The examples accept contiguous FP32 matrices of finite scores on one NVIDIA CUDA GPU, with 1–65536 columns and any nonnegative row count. Each program streams one row in bounded tiles. The code implements forward softmax without a registered backward operation.

## Concepts: why an online normalizer is useful

The word **online** means update the computation as another value or block arrives. It does not refer to network access or a training optimizer. The mathematical output is the same ordinary softmax:

$$
y_i=\frac{e^{x_i-m}}{\ell},\qquad
m=\max_j x_j,\qquad \ell=\sum_j e^{x_j-m}.
$$

The difficult part is calculating the denominator while the final maximum is still unknown. This guide derives the recurrence behind the [online normalizer algorithm](https://arxiv.org/abs/1805.02867), then implements it for blocks rather than one scalar at a time.

### 1. The three-pass streaming baseline

If a long row cannot be retained locally, a straightforward stable implementation can scan it three times:

1. Find its final maximum.
2. Read it again to sum exponentials shifted by that maximum.
3. Read it once more to calculate and write every probability.

The earlier [softmax guide](06_softmax.md) keeps a whole padded row locally and needs only one input load for that row. Here the goal is different: keep local tile storage bounded while handling rows much wider than that tile.

The online algorithm combines steps 1 and 2 into one scan. A second scan still writes the full probability vector. **An online normalizer is a one-pass statistics calculation; this standalone softmax output kernel is a two-pass algorithm.**

### 2. Two numbers summarize every processed prefix

After processing a set of positions $P$, maintain the invariant

$$
m_P=\max_{i\in P}x_i,\qquad
\ell_P=\sum_{i\in P}e^{x_i-m_P}.
$$

`m` tells us the exponential reference scale. `ell` is the total exponential mass measured at that scale. They must be interpreted together: a denominator from one maximum cannot simply be added to a denominator from another maximum.

Initialize an empty prefix with `m=-inf` and `ell=0`. The first block contains at least one finite score, so its new maximum is finite and the old contribution scales to zero. Subsequent valid-prefix denominators lie between `1` and the processed element count in exact arithmetic: at least one score equals the maximum, and every shifted exponential is at most one.

### 3. Derive the update for one new score

Given old state $(m,\ell)$ and a new score $x$, the new maximum is

$$
m'=\max(m,x).
$$

Every old exponential was measured relative to $m$. To measure it relative to $m'$, use

$$
e^{x_i-m'}=e^{x_i-m}e^{m-m'}.
$$

The factor is the same for every old input, so it can be pulled outside the old sum:

$$
\boxed{\ell'=\ell\,e^{m-m'}+e^{x-m'}}.
$$

If the maximum does not increase, the correction factor is one. If a larger score arrives, the factor is less than one and rescales all old mass at once. We do not need to revisit every earlier score to update the denominator.

### 4. Generalize the update to a block

For a new block $B$, let $m_B=\max_{i\in B}x_i$. Then

$$
\boxed{m'=\max(m,m_B)},\qquad
\boxed{\ell'=\ell\,e^{m-m'}+\sum_{i\in B}e^{x_i-m'}}.
$$

Within a block, the maximum and sum are parallel reductions. Across blocks, this teaching kernel carries the two-number state through a loop.

If the block already has its own state $(m_B,\ell_B)$, the equivalent merge rule is

$$
\ell'=\ell\,e^{m-m'}+\ell_B\,e^{m_B-m'}.
$$

This rule combines summaries of two nonempty groups. It is associative in exact arithmetic, allowing tree-based combinations as a later extension. Floating-point grouping can still change small rounding errors.

```text
old state (m, ell) ------------------------+
                                          |
new block -> block max -> new max m' ------+--> correction exp(m-m')
             |                            |             |
             +--> exp(block-m') -> sum ----+--> rescaled old ell + new mass
                                                               |
                                                        new state (m', ell')
```

### 5. A worked example where the maximum changes

Take `x=[1,2,3,4]` and blocks `[1,2]`, `[3,4]`.

| Processed prefix | Maximum | Shifted denominator |
| --- | ---: | ---: |
| `[1,2]` | 2 | $e^{-1}+1=1.367879$ |
| `[1,2,3,4]` | 4 | $e^{-3}+e^{-2}+e^{-1}+1=1.553002$ |

The second update uses `alpha=exp(2-4)=0.135335`:

$$
\ell'=1.367879\times0.135335+(e^{-1}+1)=1.553002.
$$

The final probabilities are approximately `[0.032059,0.087144,0.236883,0.643914]`. Adding the two blocks' locally shifted denominators directly would produce `2.735759`, because each local denominator uses a different maximum. That sum has the wrong scale.

### 6. Why the output needs another pass

After processing `[1,2]`, their probabilities are normalized only over that prefix. A later score of `4` changes both the maximum reference and the total denominator. Earlier probabilities cannot be considered final yet.

Once the entire row has been scanned, the final state gives

$$
y_i=e^{x_i-m_{\mathrm{final}}}/\ell_{\mathrm{final}}.
$$

This kernel rereads the row to write those values. It avoids retaining a full row of intermediate exponentials; it does not avoid allocating the requested full output matrix.

| Standalone implementation | Logical input reads per element | Output writes | Local row storage |
| --- | ---: | ---: | --- |
| Separate streaming max, sum, output passes | 3 | 1 | Bounded tiles |
| Online normalizer, then output pass | 2 | 1 | Bounded tiles + two scalars |
| Whole-row fused softmax from guide 6 | 1 | 1 | Whole padded row |

The online code also stores two statistics per row for inspection. Logical counts are not measured HBM transactions; caching and implementation details affect physical traffic and speed. A loop with two dependent running statistics can be slower than a whole-row reduction when the row fits locally.

### 7. Connection to attention and FlashAttention

Attention needs the weighted value sum, not necessarily every probability stored separately. Along with the denominator, maintain a vector numerator

$$
u=\sum_{i\in P}e^{s_i-m}v_i.
$$

When a new score block changes the maximum to $m'$, update it with the **same correction**:

$$
u'=e^{m-m'}u+\sum_{i\in B}e^{s_i-m'}v_i,\qquad o=u_{\mathrm{final}}/\ell_{\mathrm{final}}.
$$

A tiled attention algorithm can use these statistics while producing and consuming score blocks locally, avoiding a full stored probability matrix. That is a key ingredient in memory-efficient exact attention; the [FlashAttention paper](https://arxiv.org/abs/2205.14135) develops the complete attention algorithm and its memory-access analysis. This tutorial implements standalone online softmax, while guide 12 deliberately materializes both intermediate matrices.

### 8. Numerical and masking details

All exponent arguments in the update are nonpositive for valid finite inputs, which prevents exponential overflow. Very small contributions may underflow to zero, as in ordinary stable softmax. Padding is negative infinity, so padded exponentials contribute zero. Every visited block contains at least one valid finite score.

A completely empty or fully masked block would require special handling: `-inf - (-inf)` is undefined. This implementation rejects zero-width rows and accepts finite input scores, so that situation does not arise. If you extend it to arbitrary attention masks, define the treatment of fully masked blocks and rows explicitly.

## 1. Imports and example rows

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
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
x = torch.tensor([[1.,2.,3.,4.],[1001.,1002.,1003.,1004.]],device=DEVICE,dtype=DTYPE)
```

The two rows have the same probabilities despite their different absolute offsets.

## 2. PyTorch eager: the ordinary stable formula

Use the conventional formula as the reference output.

```python
maximum = x.amax(dim=-1,keepdim=True)
exponentials = torch.exp(x-maximum)
denominator = exponentials.sum(dim=-1,keepdim=True)
y_formula = exponentials / denominator
y_eager = torch.softmax(x,dim=-1)
torch.testing.assert_close(y_formula,y_eager,rtol=1e-5,atol=1e-6)
print("Final maxima:",maximum.cpu().flatten().tolist())
print("Final denominators:",denominator.cpu().flatten().tolist())
print("Probabilities:\n",y_eager.cpu())
```

The maxima are `4` and `1004`, the denominators are both approximately `1.553002`, and both probability rows match the worked example.

## 3. PyTorch functions: built-in and online recurrence

The online reference uses blockwise eager operations and returns optional statistics.

```python
def softmax_torch(x):
    return torch.softmax(x,dim=-1)


def online_softmax_torch(x,block_size=256,return_stats=False):
    if x.ndim != 2 or x.shape[1] == 0 or block_size <= 0:
        raise ValueError("Use a matrix with positive width and block size.")
    rows,columns = x.shape
    maximum = torch.full((rows,1),-float("inf"),device=x.device,dtype=x.dtype)
    denominator = torch.zeros_like(maximum)
    for start in range(0,columns,block_size):
        chunk = x[:,start:start+block_size]
        new_maximum = torch.maximum(maximum,chunk.amax(dim=-1,keepdim=True))
        correction = torch.exp(maximum-new_maximum)
        denominator = denominator*correction + torch.exp(chunk-new_maximum).sum(dim=-1,keepdim=True)
        maximum = new_maximum
    output = torch.empty_like(x)
    for start in range(0,columns,block_size):
        output[:,start:start+block_size] = torch.exp(x[:,start:start+block_size]-maximum)/denominator
    return (output,maximum.squeeze(-1),denominator.squeeze(-1)) if return_stats else output


y_online,m_online,l_online = online_softmax_torch(x,block_size=2,return_stats=True)
torch.testing.assert_close(y_online,y_eager,rtol=1e-5,atol=1e-6)
print("Online maxima:",m_online.cpu().tolist())
print("Online denominators:",l_online.cpu().tolist())
```

This reference makes the recurrence visible, but its Python loops submit many eager operations. It is an algorithm demonstration, not a replacement performance baseline for the built-in softmax.

## 4. Triton kernel: stream statistics, then write probabilities

Each program keeps a tile and two running FP32 scalars.

```python
@triton.jit
def online_softmax_kernel(
    X, Y, Maxima, Denominators,
    N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    local_columns = tl.arange(0,BLOCK_SIZE)
    running_max = tl.full((),-float("inf"),tl.float32)
    running_sum = tl.full((),0.0,tl.float32)

    # Pass 1: update the maximum and shifted denominator together.
    for start in range(0,N_COLS,BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(X + row*N_COLS + columns,
                         mask=columns<N_COLS,other=-float("inf"))
        block_max = tl.max(values,axis=0)
        new_max = tl.maximum(running_max,block_max)
        correction = tl.exp(running_max-new_max)
        block_sum = tl.sum(tl.exp(values-new_max),axis=0)
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    tl.store(Maxima + row,running_max)
    tl.store(Denominators + row,running_sum)

    # Pass 2: final normalization needs the completed row statistics.
    for start in range(0,N_COLS,BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(X + row*N_COLS + columns,
                         mask=columns<N_COLS,other=-float("inf"))
        probabilities = tl.exp(values-running_max) / running_sum
        tl.store(Y + row*N_COLS + columns,probabilities,mask=columns<N_COLS)
```

1. Initialize `running_max=-inf` and `running_sum=0`.
2. Read a block and reduce its maximum.
3. Compute the new common maximum and rescale the old denominator.
4. Add this block's exponential mass at the new scale.
5. Save the completed statistics and reread blocks to write final probabilities.

Both passes are inside one kernel launch. `BLOCK_SIZE` bounds the logical local tile even when the full row is much wider.

## 5. Python wrapper: choose a bounded tile size

Tile sizes do not need to equal or exceed the row width.

```python
def online_softmax_triton(x,block_size=256,return_stats=False):
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("Input must be a contiguous FP32 CUDA matrix.")
    rows,columns = x.shape
    if not 1 <= columns <= 65536:
        raise ValueError("This teaching example supports 1–65536 columns.")
    if block_size not in (128,256,512,1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")
    output = torch.empty_like(x)
    maxima = torch.empty(rows,device=x.device,dtype=torch.float32)
    denominators = torch.empty_like(maxima)
    if rows > 0:
        with torch.cuda.device(x.device):
            online_softmax_kernel[(rows,)](
                x,output,maxima,denominators,N_COLS=columns,BLOCK_SIZE=block_size,num_warps=4,
            )
    return (output,maxima,denominators) if return_stats else output
```

The width bound is a practical teaching limit, not a mathematical limit of the recurrence. Zero-row batches return empty tensors. Full output storage still scales with the number of input elements, while local tile storage stays bounded.

## 6. Match values and verify the running-statistic invariants

Check the denominator itself, not just the final probabilities.

```python
actual,saved_m,saved_l = online_softmax_triton(x,return_stats=True)
torch.testing.assert_close(actual,y_eager,rtol=1e-5,atol=1e-6)
print("PyTorch:",y_eager.cpu().tolist())
print("Triton: ",actual.cpu().tolist())
print("Maxima:",saved_m.cpu().tolist(),"denominators:",saved_l.cpu().tolist())
for rows,columns in ((0,7),(1,1),(3,7),(2,257),(4,1003),(2,4097),(2,65535)):
    values = torch.randn(rows,columns,device=DEVICE,dtype=DTYPE)
    if rows:
        values[:,-1] += 10.0  # A late larger maximum forces rescaling.
    expected = softmax_torch(values)
    expected_m = values.amax(dim=-1)
    expected_l = torch.exp(values-expected_m[:,None]).sum(dim=-1)
    for block_size in (128,256,512,1024):
        result,m,l = online_softmax_triton(values,block_size,return_stats=True)
        torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(m,expected_m,rtol=0,atol=0)
        torch.testing.assert_close(l,expected_l,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(result.sum(-1),torch.ones(rows,device=DEVICE,dtype=DTYPE),rtol=1e-5,atol=1e-6)
print("All online-state, late-maximum, tail, row-sum, and wide-row checks passed.")
```

A late maximum catches a missing correction factor. Different tile sizes change grouping and can produce small floating-point differences without changing the intended softmax.

## 7. Compare built-in and online implementations

Use the built-in function as the speedup baseline, and time the blockwise eager algorithm separately.

```python
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("shape       eager us  function us  online torch us  online Triton us  speedup  max error")
for rows,columns in ((16,1024),(32,8192),(8,32768),(4,65536)):
    values = torch.randn(rows,columns,device=DEVICE,dtype=DTYPE)
    expected = softmax_torch(values)
    actual = online_softmax_triton(values)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.softmax(values,dim=-1))
    p = bench_ms(lambda: softmax_torch(values))
    o = bench_ms(lambda: online_softmax_torch(values))
    t = bench_ms(lambda: online_softmax_triton(values))
    print(f"{rows}x{columns} {e*1000:9.3f} {p*1000:12.3f} {o*1000:15.3f} {t*1000:17.3f} {p/t:7.2f}x {error:.3e}")
```

Each timed call allocates its output. Input creation, first compilation, assertions, CPU copies, and printing are outside timing. [`do_bench`](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html) returns milliseconds; the table prints median microseconds. GPU event timings can include host dispatch gaps and are separate from full CPU wall-clock time. A speedup above `1.0x` means the Triton measurement was faster. Run the code for actual timings; no speedup is assumed. Triton includes both passes and its statistic buffers. With only a few rows, this one-program-per-row design exposes limited parallelism; smaller tiles also increase the number of dependent loop iterations.

## 8. Trace a changing maximum with ordinary Python numbers

This diagnostic runs on the CPU and prints the state after every block.

```python
import math


def trace_online_row(values,block_size=2):
    maximum,denominator = -math.inf,0.0
    for start in range(0,len(values),block_size):
        chunk = values[start:start+block_size]
        new_maximum = max(maximum,max(chunk))
        correction = math.exp(maximum-new_maximum)
        new_mass = sum(math.exp(value-new_maximum) for value in chunk)
        denominator = denominator*correction + new_mass
        maximum = new_maximum
        print(f"block={chunk}, max={maximum:.6f}, correction={correction:.6f}, denominator={denominator:.6f}")
    return [math.exp(value-maximum)/denominator for value in values]


traced = trace_online_row([1.,2.,3.,4.])
print("Final probabilities:",traced)
reference = torch.softmax(torch.tensor([1.,2.,3.,4.],dtype=torch.float64),dim=-1)
torch.testing.assert_close(torch.tensor(traced,dtype=torch.float64),reference,rtol=1e-12,atol=1e-12)
```

The first correction is zero, and the second is `exp(-2)`. Reverse the block order to see a correction of one when the maximum is already known; the final probabilities still follow the same equation.

## Try next

Try changing block sizes and moving the largest value between the first and last blocks. Continue with [FlashAttention forward](14_flashattention_forward.md), which carries the weighted value numerator together with the denominator to compute attention without storing the full probability matrix.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all preceding code: imports, references, kernels, launch helpers, correctness checks, and benchmarks. Comments mark each step.

```python
# 1. Imports and example rows
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
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))
x = torch.tensor([[1.,2.,3.,4.],[1001.,1002.,1003.,1004.]],device=DEVICE,dtype=DTYPE)


# 2. PyTorch eager: the ordinary stable formula
maximum = x.amax(dim=-1,keepdim=True)
exponentials = torch.exp(x-maximum)
denominator = exponentials.sum(dim=-1,keepdim=True)
y_formula = exponentials / denominator
y_eager = torch.softmax(x,dim=-1)
torch.testing.assert_close(y_formula,y_eager,rtol=1e-5,atol=1e-6)
print("Final maxima:",maximum.cpu().flatten().tolist())
print("Final denominators:",denominator.cpu().flatten().tolist())
print("Probabilities:\n",y_eager.cpu())


# 3. PyTorch functions: built-in and online recurrence
def softmax_torch(x):
    return torch.softmax(x,dim=-1)


def online_softmax_torch(x,block_size=256,return_stats=False):
    if x.ndim != 2 or x.shape[1] == 0 or block_size <= 0:
        raise ValueError("Use a matrix with positive width and block size.")
    rows,columns = x.shape
    maximum = torch.full((rows,1),-float("inf"),device=x.device,dtype=x.dtype)
    denominator = torch.zeros_like(maximum)
    for start in range(0,columns,block_size):
        chunk = x[:,start:start+block_size]
        new_maximum = torch.maximum(maximum,chunk.amax(dim=-1,keepdim=True))
        correction = torch.exp(maximum-new_maximum)
        denominator = denominator*correction + torch.exp(chunk-new_maximum).sum(dim=-1,keepdim=True)
        maximum = new_maximum
    output = torch.empty_like(x)
    for start in range(0,columns,block_size):
        output[:,start:start+block_size] = torch.exp(x[:,start:start+block_size]-maximum)/denominator
    return (output,maximum.squeeze(-1),denominator.squeeze(-1)) if return_stats else output


y_online,m_online,l_online = online_softmax_torch(x,block_size=2,return_stats=True)
torch.testing.assert_close(y_online,y_eager,rtol=1e-5,atol=1e-6)
print("Online maxima:",m_online.cpu().tolist())
print("Online denominators:",l_online.cpu().tolist())


# 4. Triton kernel: stream statistics, then write probabilities
@triton.jit
def online_softmax_kernel(
    X, Y, Maxima, Denominators,
    N_COLS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    local_columns = tl.arange(0,BLOCK_SIZE)
    running_max = tl.full((),-float("inf"),tl.float32)
    running_sum = tl.full((),0.0,tl.float32)

    # Pass 1: update the maximum and shifted denominator together.
    for start in range(0,N_COLS,BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(X + row*N_COLS + columns,
                         mask=columns<N_COLS,other=-float("inf"))
        block_max = tl.max(values,axis=0)
        new_max = tl.maximum(running_max,block_max)
        correction = tl.exp(running_max-new_max)
        block_sum = tl.sum(tl.exp(values-new_max),axis=0)
        running_sum = running_sum * correction + block_sum
        running_max = new_max

    tl.store(Maxima + row,running_max)
    tl.store(Denominators + row,running_sum)

    # Pass 2: final normalization needs the completed row statistics.
    for start in range(0,N_COLS,BLOCK_SIZE):
        columns = start + local_columns
        values = tl.load(X + row*N_COLS + columns,
                         mask=columns<N_COLS,other=-float("inf"))
        probabilities = tl.exp(values-running_max) / running_sum
        tl.store(Y + row*N_COLS + columns,probabilities,mask=columns<N_COLS)


# 5. Python wrapper: choose a bounded tile size
def online_softmax_triton(x,block_size=256,return_stats=False):
    if not x.is_cuda or x.ndim != 2 or not x.is_contiguous() or x.dtype != torch.float32:
        raise ValueError("Input must be a contiguous FP32 CUDA matrix.")
    rows,columns = x.shape
    if not 1 <= columns <= 65536:
        raise ValueError("This teaching example supports 1–65536 columns.")
    if block_size not in (128,256,512,1024):
        raise ValueError("Use a block size of 128, 256, 512, or 1024.")
    output = torch.empty_like(x)
    maxima = torch.empty(rows,device=x.device,dtype=torch.float32)
    denominators = torch.empty_like(maxima)
    if rows > 0:
        with torch.cuda.device(x.device):
            online_softmax_kernel[(rows,)](
                x,output,maxima,denominators,N_COLS=columns,BLOCK_SIZE=block_size,num_warps=4,
            )
    return (output,maxima,denominators) if return_stats else output


# 6. Match values and verify the running-statistic invariants
actual,saved_m,saved_l = online_softmax_triton(x,return_stats=True)
torch.testing.assert_close(actual,y_eager,rtol=1e-5,atol=1e-6)
print("PyTorch:",y_eager.cpu().tolist())
print("Triton: ",actual.cpu().tolist())
print("Maxima:",saved_m.cpu().tolist(),"denominators:",saved_l.cpu().tolist())
for rows,columns in ((0,7),(1,1),(3,7),(2,257),(4,1003),(2,4097),(2,65535)):
    values = torch.randn(rows,columns,device=DEVICE,dtype=DTYPE)
    if rows:
        values[:,-1] += 10.0  # A late larger maximum forces rescaling.
    expected = softmax_torch(values)
    expected_m = values.amax(dim=-1)
    expected_l = torch.exp(values-expected_m[:,None]).sum(dim=-1)
    for block_size in (128,256,512,1024):
        result,m,l = online_softmax_triton(values,block_size,return_stats=True)
        torch.testing.assert_close(result,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(m,expected_m,rtol=0,atol=0)
        torch.testing.assert_close(l,expected_l,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(result.sum(-1),torch.ones(rows,device=DEVICE,dtype=DTYPE),rtol=1e-5,atol=1e-6)
print("All online-state, late-maximum, tail, row-sum, and wide-row checks passed.")


# 7. Compare built-in and online implementations
def bench_ms(fn):
    # Compile and warm up before steady-state GPU event timing.
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

print("shape       eager us  function us  online torch us  online Triton us  speedup  max error")
for rows,columns in ((16,1024),(32,8192),(8,32768),(4,65536)):
    values = torch.randn(rows,columns,device=DEVICE,dtype=DTYPE)
    expected = softmax_torch(values)
    actual = online_softmax_triton(values)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    error = (actual-expected).abs().max().item()
    e = bench_ms(lambda: torch.softmax(values,dim=-1))
    p = bench_ms(lambda: softmax_torch(values))
    o = bench_ms(lambda: online_softmax_torch(values))
    t = bench_ms(lambda: online_softmax_triton(values))
    print(f"{rows}x{columns} {e*1000:9.3f} {p*1000:12.3f} {o*1000:15.3f} {t*1000:17.3f} {p/t:7.2f}x {error:.3e}")


# 8. Trace a changing maximum with ordinary Python numbers
import math


def trace_online_row(values,block_size=2):
    maximum,denominator = -math.inf,0.0
    for start in range(0,len(values),block_size):
        chunk = values[start:start+block_size]
        new_maximum = max(maximum,max(chunk))
        correction = math.exp(maximum-new_maximum)
        new_mass = sum(math.exp(value-new_maximum) for value in chunk)
        denominator = denominator*correction + new_mass
        maximum = new_maximum
        print(f"block={chunk}, max={maximum:.6f}, correction={correction:.6f}, denominator={denominator:.6f}")
    return [math.exp(value-maximum)/denominator for value in values]


traced = trace_online_row([1.,2.,3.,4.])
print("Final probabilities:",traced)
reference = torch.softmax(torch.tensor([1.,2.,3.,4.],dtype=torch.float64),dim=-1)
torch.testing.assert_close(torch.tensor(traced,dtype=torch.float64),reference,rtol=1e-12,atol=1e-12)
```
