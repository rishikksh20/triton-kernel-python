# 24. SwiGLU: fuse two matrix projections with a gated epilogue

Previous: [Dropout](23_dropout.md). Next: [Cross entropy](25_cross_entropy.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. X is contiguous FP32 `(M,K)` and gate/up weights are contiguous FP32 `(K,N)` on one NVIDIA CUDA GPU. Dimensions are positive and values are finite with moderate products/sums. The output is FP32 `(M,N)`. IEEE FP32 dot products emphasize numerical clarity; this forward-only kernel has no registered backward operation.

## Concepts: multiply an activated gate by a second projection

SwiGLU uses two independent projections of the same input:

$$
G=XW_g,\qquad U=XW_u,\qquad
\operatorname{SiLU}(z)=z\sigma(z),\qquad Y=\operatorname{SiLU}(G)\odot U.
$$

The [GLU variants paper](https://arxiv.org/abs/2002.05202) describes gated activation variants including SwiGLU. This notebook implements the gate/up stage without bias. A full feed-forward layer also applies a down projection `Y @ W_down`, shown separately in section 8.

### 1. Reuse each input tile across both projections

A program owns a `(BLOCK_M,BLOCK_N)` output tile. It loads one X tile and two weight tiles per K iteration. Two FP32 accumulators collect G and U. After the complete K reduction, compute SiLU(G) * U and store only Y.

```text
                  W_gate tile -> gate accumulator -> SiLU --+
X tile, reused ---+                                          * -> Y tile
                  W_up tile   -> up accumulator ------------+
```

### 2. The activation belongs after the K reduction

Applying SiLU to a partial gate accumulator and summing the results does not equal SiLU of the final projection. Fusion here combines two complete matrix products with their output epilogue; it does not move a nonlinear operation through a reduction.

### 3. Fusion saves intermediates but increases live state

The kernel avoids storing and reloading full G and U matrices. It retains two accumulator tiles and two weight tiles, increasing register pressure. A tuned library implementation with separate GEMMs can still win. Native FP16/BF16 Tensor Core paths and tile tuning are separate accuracy/performance choices; see [matmul](09_matmul.md) and [fused matmul epilogues](10_matmul_bias_activation.md).

Negative gate values do not simply become zero: SiLU is smooth and can be negative. Testing both signs distinguishes SwiGLU from ReGLU.

## 1. Imports and a small gated projection

```python
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))

x = torch.tensor([[1., -1., 2.], [0., 2., -1.]], device=DEVICE)
wg = torch.tensor([[1., -1.], [0., 1.], [-1., 0.]], device=DEVICE)
wu = torch.tensor([[2., 1.], [1., -1.], [0., 2.]], device=DEVICE)
```

The two weight matrices have matching shapes but different values.

## 2. PyTorch eager: inspect gate, up projection, and product

```python
gate, up = x @ wg, x @ wu
y_eager = F.silu(gate) * up
print("Gate:\n", gate.cpu())
print("Up:\n", up.cpu())
print("SwiGLU:\n", y_eager.cpu())
```

Printing intermediate projections makes the operand roles explicit.

## 3. PyTorch function: the reusable SwiGLU stage

```python
def swiglu_torch(x, wg, wu):
    return F.silu(x @ wg) * (x @ wu)
```

The down projection, biases, residual connection, and normalization are outside this operation.

## 4. Triton: two accumulators and one fused epilogue

```python
@triton.jit
def swiglu_kernel(X, WG, WU, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(0, tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        offsets = k[:, None] * N + cols[None, :]
        mask = (k[:, None] < K) & (cols[None, :] < N)
        g = tl.load(WG + offsets, mask, other=0)
        u = tl.load(WU + offsets, mask, other=0)
        gate = tl.dot(a, g, gate, input_precision="ieee")
        up = tl.dot(a, u, up, input_precision="ieee")
    result = (gate / (1 + tl.exp(-gate))) * up
    tl.store(Y + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))
```

Only the completed accumulators feed the activation. Each output tile has one writer.

## 5. Python wrapper: check shapes and launch output tiles

```python
def swiglu_triton(x, wg, wu):
    if x.ndim != 2 or wg.ndim != 2 or wu.shape != wg.shape or x.shape[1] != wg.shape[0]:
        raise ValueError("require X (M,K) and matching gate/up weights (K,N)")
    m, k = x.shape
    n = wg.shape[1]
    if min(m, n, k) < 1 or max(x.numel(), wg.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions and 32-bit-safe offsets")
    if not x.is_cuda or any(t.dtype != torch.float32 or t.device != x.device or not t.is_contiguous() for t in (x, wg, wu)):
        raise ValueError("all inputs must be contiguous FP32 on the same CUDA device")
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    swiglu_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        x, wg, wu, out, m, n, k, 32, 32, 32, num_warps=4)
    return out
```

This fixed configuration reuses an X tile for both projections. It is not an autotuned Tensor Core implementation.

## 6. Compare gate signs and all three tile boundaries

```python
torch.testing.assert_close(swiglu_triton(x, wg, wu), y_eager, rtol=1e-4, atol=1e-4)
for m, n, k in ((1, 1, 1), (17, 19, 23), (33, 65, 31), (64, 33, 65)):
    sample = torch.randn(m, k, device=DEVICE)
    gate_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    up_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    actual = swiglu_triton(sample, gate_weights, up_weights)
    expected = swiglu_torch(sample, gate_weights, up_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
zero_gate = torch.zeros_like(wg)
torch.testing.assert_close(swiglu_triton(x, zero_gate, wu), torch.zeros_like(y_eager), rtol=0, atol=0)
print("SwiGLU values, zero gate, and M/N/K tail checks passed.")
```

Weight normalization keeps the random test projections moderate as K changes. The tolerances are not a guarantee for arbitrarily large inputs.

## 7. Benchmark the complete gate/up stage

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for m, n, k in ((64, 128, 64), (128, 256, 128), (256, 256, 256)):
    sample = torch.randn(m, k, device=DEVICE)
    gate_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    up_weights = torch.randn_like(gate_weights) / k**0.5
    actual = swiglu_triton(sample, gate_weights, up_weights)
    expected = swiglu_torch(sample, gate_weights, up_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    p = bench_ms(lambda: swiglu_torch(sample, gate_weights, up_weights))
    t = bench_ms(lambda: swiglu_triton(sample, gate_weights, up_weights))
    print(f"{m,n,k}: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, {p/t:.2f}x")
```

Both paths include two projections, SiLU, multiplication, and output allocation. The Triton path avoids full gate/up intermediates. Compilation, checks, and printing are outside GPU event timing. No speedup is assumed.

## 8. Add the down projection to form a feed-forward stage

```python
w_down = torch.tensor([[1., 2., 0.], [-1., 0., 1.]], device=DEVICE)
full_reference = swiglu_torch(x, wg, wu) @ w_down
full_output = swiglu_triton(x, wg, wu) @ w_down
torch.testing.assert_close(full_output, full_reference, rtol=1e-4, atol=1e-4)
print("After the separate down projection:\n", full_output.cpu())
```

The down projection is a separate PyTorch GEMM. Fusing it would require a new schedule because it reduces across N, the feature dimension of this tutorial’s output.

## Try next

Continue with [cross entropy](25_cross_entropy.md). Try lower-precision inputs or bias epilogues, and assess whether the extra live accumulator state offsets the saved memory traffic.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a small gated projection
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))

x = torch.tensor([[1., -1., 2.], [0., 2., -1.]], device=DEVICE)
wg = torch.tensor([[1., -1.], [0., 1.], [-1., 0.]], device=DEVICE)
wu = torch.tensor([[2., 1.], [1., -1.], [0., 2.]], device=DEVICE)


# 2. PyTorch eager: inspect gate, up projection, and product
gate, up = x @ wg, x @ wu
y_eager = F.silu(gate) * up
print("Gate:\n", gate.cpu())
print("Up:\n", up.cpu())
print("SwiGLU:\n", y_eager.cpu())


# 3. PyTorch function: the reusable SwiGLU stage
def swiglu_torch(x, wg, wu):
    return F.silu(x @ wg) * (x @ wu)


# 4. Triton: two accumulators and one fused epilogue
@triton.jit
def swiglu_kernel(X, WG, WU, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    gate = tl.zeros((BM, BN), tl.float32)
    up = tl.zeros((BM, BN), tl.float32)
    for block in range(0, tl.cdiv(K, BK)):
        k = block * BK + kk
        a = tl.load(X + rows[:, None] * K + k[None, :],
                    (rows[:, None] < M) & (k[None, :] < K), other=0)
        offsets = k[:, None] * N + cols[None, :]
        mask = (k[:, None] < K) & (cols[None, :] < N)
        g = tl.load(WG + offsets, mask, other=0)
        u = tl.load(WU + offsets, mask, other=0)
        gate = tl.dot(a, g, gate, input_precision="ieee")
        up = tl.dot(a, u, up, input_precision="ieee")
    result = (gate / (1 + tl.exp(-gate))) * up
    tl.store(Y + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))


# 5. Python wrapper: check shapes and launch output tiles
def swiglu_triton(x, wg, wu):
    if x.ndim != 2 or wg.ndim != 2 or wu.shape != wg.shape or x.shape[1] != wg.shape[0]:
        raise ValueError("require X (M,K) and matching gate/up weights (K,N)")
    m, k = x.shape
    n = wg.shape[1]
    if min(m, n, k) < 1 or max(x.numel(), wg.numel(), m*n) >= 2**31:
        raise ValueError("require positive dimensions and 32-bit-safe offsets")
    if not x.is_cuda or any(t.dtype != torch.float32 or t.device != x.device or not t.is_contiguous() for t in (x, wg, wu)):
        raise ValueError("all inputs must be contiguous FP32 on the same CUDA device")
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    swiglu_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
        x, wg, wu, out, m, n, k, 32, 32, 32, num_warps=4)
    return out


# 6. Compare gate signs and all three tile boundaries
torch.testing.assert_close(swiglu_triton(x, wg, wu), y_eager, rtol=1e-4, atol=1e-4)
for m, n, k in ((1, 1, 1), (17, 19, 23), (33, 65, 31), (64, 33, 65)):
    sample = torch.randn(m, k, device=DEVICE)
    gate_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    up_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    actual = swiglu_triton(sample, gate_weights, up_weights)
    expected = swiglu_torch(sample, gate_weights, up_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
zero_gate = torch.zeros_like(wg)
torch.testing.assert_close(swiglu_triton(x, zero_gate, wu), torch.zeros_like(y_eager), rtol=0, atol=0)
print("SwiGLU values, zero gate, and M/N/K tail checks passed.")


# 7. Benchmark the complete gate/up stage
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for m, n, k in ((64, 128, 64), (128, 256, 128), (256, 256, 256)):
    sample = torch.randn(m, k, device=DEVICE)
    gate_weights = torch.randn(k, n, device=DEVICE) / k**0.5
    up_weights = torch.randn_like(gate_weights) / k**0.5
    actual = swiglu_triton(sample, gate_weights, up_weights)
    expected = swiglu_torch(sample, gate_weights, up_weights)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-3)
    p = bench_ms(lambda: swiglu_torch(sample, gate_weights, up_weights))
    t = bench_ms(lambda: swiglu_triton(sample, gate_weights, up_weights))
    print(f"{m,n,k}: PyTorch {p*1000:.3f} us, Triton {t*1000:.3f} us, {p/t:.2f}x")


# 8. Add the down projection to form a feed-forward stage
w_down = torch.tensor([[1., 2., 0.], [-1., 0., 1.]], device=DEVICE)
full_reference = swiglu_torch(x, wg, wu) @ w_down
full_output = swiglu_triton(x, wg, wu) @ w_down
torch.testing.assert_close(full_output, full_reference, rtol=1e-4, atol=1e-4)
print("After the separate down projection:\n", full_output.cpu())
```
