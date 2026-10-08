# 23. Dropout: reproducible random masks and a regenerated backward pass

Previous: [Decode attention](22_decode_attention.md). Next: [SwiGLU](24_swiglu.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Inputs are finite, moderate, contiguous FP32 tensors on one NVIDIA CUDA GPU with fewer than 2**31 elements. Drop probability p lies in `[0,1]`; a caller-supplied nonnegative INT32-range seed identifies the mask. The direct kernel is forward-only, and section 8 adds a custom first-order PyTorch autograd operation.

## Concepts: preserve the expected activation

For inverted training dropout with p < 1, draw a uniform random u_i per element:

$$
b_i=\mathbf1[u_i\ge p],\qquad y_i=\frac{b_i x_i}{1-p},\qquad
\frac{\partial\mathcal L}{\partial x_i}=\frac{b_i}{1-p}\frac{\partial\mathcal L}{\partial y_i}.
$$

The survivors scale up so that their expectation equals the input in the ideal continuous random model. Evaluation mode is the identity. This guide handles p=1 with an explicit all-zero branch, avoiding division by zero.

### 1. Associate randomness with the logical element index

`tl.rand(seed, offset)` returns an FP32 uniform value for each offset. Use the tensor's flattened global element index, not a thread-local or tile-local index. Then changing the block size preserves the logical mask. The [Triton random API](https://triton-lang.org/main/python-api/generated/triton.language.rand.html) describes this interface.

### 2. Reuse a seed only when the same mask is intended

A seed is an explicit input. The operation does not advance PyTorch's random generator. Training code must provide new seeds for independent invocations/layers; reusing a fixed seed repeats the mask. Checkpoint recomputation and backward deliberately use the original seed and element offsets.

The [official seeded-dropout tutorial](https://triton-lang.org/main/getting-started/tutorials/04-low-memory-dropout.html) explains how deterministic regeneration reduces saved-state storage. We optionally expose the mask for correctness tests, while the normal path allocates only an output.

### 3. Compare arithmetic using the same mask

PyTorch and Triton need not map the same numeric seed to the same random tensor. Comparing two independently generated outputs elementwise is therefore invalid. First expose the Triton mask, then apply that mask in PyTorch. Separately check repeated-seed determinism, launch-size independence, and the empirical kept fraction.

```text
(seed, global offsets) -> uniform samples -> keep decisions
x -> scale retained elements -> y
backward: regenerate those decisions -> scale upstream gradient
```

The custom autograd wrapper saves probability, effective training mode, and seed as scalar metadata. It saves neither x nor a full mask. It supports first-order gradients only.

## 1. Imports and a short activation vector

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

x = torch.arange(1, 18, device=DEVICE).float()
p, seed = 0.25, 123
```

Nonzero inputs make kept and dropped elements easy to see.

## 2. PyTorch eager: inverted dropout with an explicit mask

```python
keep = torch.rand_like(x) >= p
y_eager = torch.where(keep, x / (1 - p), 0.)
print("Input:", x.cpu())
print("Explicit-mask result:", y_eager.cpu())
```

This random mask is an explanatory PyTorch sample, not the expected mask for the Triton seed.

## 3. PyTorch function: reuse a supplied mask

```python
def dropout_torch(x, keep, p):
    if p == 1:
        return torch.zeros_like(x)
    return torch.where(keep, x / (1 - p), 0.)
```

The mask reference isolates the scale and selection arithmetic from random-number generation.

## 4. Triton: stateless mask generation and endpoint branches

```python
@triton.jit
def dropout_kernel(X, Y, KEEP, N: tl.constexpr, P: tl.constexpr, SEED,
                   SAVE_MASK: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0)
    if P == 0:
        keep = tl.full((BLOCK,), True, tl.int1)
        y = x
    elif P == 1:
        keep = tl.full((BLOCK,), False, tl.int1)
        y = tl.zeros((BLOCK,), tl.float32)
    else:
        keep = tl.rand(SEED, offset) >= P
        y = tl.where(keep, x / (1 - P), 0.)
    tl.store(Y + offset, y, offset < N)
    if SAVE_MASK:
        tl.store(KEEP + offset, keep, offset < N)
```

p=0 and p=1 have dedicated paths. Padded lanes can generate random values but cannot write outside the tensor.

## 5. Python wrapper: make seed and training mode explicit

```python
def dropout_triton(x, p, seed, training=True, return_mask=False, block=256):
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or x.numel() >= 2**31:
        raise ValueError("require contiguous CUDA FP32 input with fewer than 2**31 elements")
    if not 0 <= p <= 1 or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("require p in [0,1] and an INT32-range nonnegative integer seed")
    if block not in (128, 256, 512, 1024):
        raise ValueError("block must be 128, 256, 512, or 1024")
    effective_p = float(p) if training else 0.0
    out = torch.empty_like(x)
    keep = torch.empty_like(x, dtype=torch.bool) if return_mask else out
    if x.numel():
        dropout_kernel[(triton.cdiv(x.numel(), block),)](
            x, out, keep, x.numel(), effective_p, seed, return_mask, block, num_warps=4)
    return (out, keep) if return_mask else out
```

The dummy KEEP pointer is never accessed when SAVE_MASK is false. Empty tensors return without a launch; evaluation mode copies the input into a fresh output.

## 6. Check the exact arithmetic, seeds, shapes, and launch sizes

```python
actual, keep = dropout_triton(x, p, seed, return_mask=True)
torch.testing.assert_close(actual, dropout_torch(x, keep, p), rtol=1e-6, atol=1e-6)
for shape in ((0,), (1,), (17,), (3, 257)):
    sample = torch.randn(shape, device=DEVICE)
    for probability in (0.0, 0.25, 1.0):
        actual, mask = dropout_triton(sample, probability, seed, return_mask=True)
        torch.testing.assert_close(actual, dropout_torch(sample, mask, probability), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual, dropout_triton(sample, probability, seed), rtol=0, atol=0)
        torch.testing.assert_close(actual, dropout_triton(sample, probability, seed, block=512), rtol=0, atol=0)
        torch.testing.assert_close(dropout_triton(sample, probability, seed, training=False), sample, rtol=0, atol=0)
sample = torch.ones(65536, device=DEVICE)
_, mask = dropout_triton(sample, 0.25, seed, return_mask=True)
_, different_mask = dropout_triton(sample, 0.25, seed + 1, return_mask=True)
assert not torch.equal(mask, different_mask)
assert abs(mask.float().mean().item() - 0.75) < 0.02
print("Mask arithmetic, endpoint, determinism, and empirical-rate checks passed.")
```

The kept-rate test is a broad statistical sanity check for this fixed sample, not a proof of a random generator’s quality.

## 7. Benchmark a fused RNG path and its arithmetic baseline

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(2**20, device=DEVICE)
_, mask = dropout_triton(sample, p, seed, return_mask=True)
arithmetic_ms = bench_ms(lambda: dropout_torch(sample, mask, p))
triton_ms = bench_ms(lambda: dropout_triton(sample, p, seed))
native_ms = bench_ms(lambda: F.dropout(sample, p=p, training=True))
print(f"PyTorch supplied-mask arithmetic: {arithmetic_ms*1000:.3f} us")
print(f"PyTorch native dropout with RNG:  {native_ms*1000:.3f} us")
print(f"Triton seeded dropout with RNG:   {triton_ms*1000:.3f} us")
```

The supplied-mask path excludes RNG. The other two include RNG and allocate outputs, but produce different masks. A fixed Triton seed permits repeatable timing; actual training must manage seed advancement. Compilation, checks, and printing are outside timing.

## 8. Register a first-order backward that regenerates the mask

```python
from torch.autograd.function import once_differentiable


class SeededDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed, training):
        ctx.p = float(p) if training else 0.0
        ctx.seed = seed
        return dropout_triton(x, p, seed, training)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        grad_x = dropout_triton(grad_out.contiguous(), ctx.p, ctx.seed)
        return grad_x, None, None, None


for probability, training in ((0.25, True), (0.0, True), (1.0, True), (0.25, False)):
    sample = torch.randn(3, 17, device=DEVICE, requires_grad=True)
    upstream = torch.randn_like(sample)
    out = SeededDropout.apply(sample, probability, seed, training)
    out.backward(upstream)
    effective = probability if training else 0.0
    _, mask = dropout_triton(sample.detach(), effective, seed, return_mask=True)
    expected_grad = dropout_torch(upstream, mask, effective)
    torch.testing.assert_close(sample.grad, expected_grad, rtol=1e-6, atol=1e-6)
print("Regenerated first-order backward checks passed.")
```

The derivative applies the original mask to the upstream gradient, not a newly chosen random mask. Noncontiguous upstream gradients are made contiguous before launching.

## Try next

Continue with [SwiGLU](24_swiglu.md). Try combining bias, activation, and dropout in one epilogue while preserving explicit seed ownership and the matching backward rule.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and a short activation vector
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

x = torch.arange(1, 18, device=DEVICE).float()
p, seed = 0.25, 123


# 2. PyTorch eager: inverted dropout with an explicit mask
keep = torch.rand_like(x) >= p
y_eager = torch.where(keep, x / (1 - p), 0.)
print("Input:", x.cpu())
print("Explicit-mask result:", y_eager.cpu())


# 3. PyTorch function: reuse a supplied mask
def dropout_torch(x, keep, p):
    if p == 1:
        return torch.zeros_like(x)
    return torch.where(keep, x / (1 - p), 0.)


# 4. Triton: stateless mask generation and endpoint branches
@triton.jit
def dropout_kernel(X, Y, KEEP, N: tl.constexpr, P: tl.constexpr, SEED,
                   SAVE_MASK: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offset, offset < N, other=0)
    if P == 0:
        keep = tl.full((BLOCK,), True, tl.int1)
        y = x
    elif P == 1:
        keep = tl.full((BLOCK,), False, tl.int1)
        y = tl.zeros((BLOCK,), tl.float32)
    else:
        keep = tl.rand(SEED, offset) >= P
        y = tl.where(keep, x / (1 - P), 0.)
    tl.store(Y + offset, y, offset < N)
    if SAVE_MASK:
        tl.store(KEEP + offset, keep, offset < N)


# 5. Python wrapper: make seed and training mode explicit
def dropout_triton(x, p, seed, training=True, return_mask=False, block=256):
    if x.dtype != torch.float32 or not x.is_cuda or not x.is_contiguous() or x.numel() >= 2**31:
        raise ValueError("require contiguous CUDA FP32 input with fewer than 2**31 elements")
    if not 0 <= p <= 1 or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("require p in [0,1] and an INT32-range nonnegative integer seed")
    if block not in (128, 256, 512, 1024):
        raise ValueError("block must be 128, 256, 512, or 1024")
    effective_p = float(p) if training else 0.0
    out = torch.empty_like(x)
    keep = torch.empty_like(x, dtype=torch.bool) if return_mask else out
    if x.numel():
        dropout_kernel[(triton.cdiv(x.numel(), block),)](
            x, out, keep, x.numel(), effective_p, seed, return_mask, block, num_warps=4)
    return (out, keep) if return_mask else out


# 6. Check the exact arithmetic, seeds, shapes, and launch sizes
actual, keep = dropout_triton(x, p, seed, return_mask=True)
torch.testing.assert_close(actual, dropout_torch(x, keep, p), rtol=1e-6, atol=1e-6)
for shape in ((0,), (1,), (17,), (3, 257)):
    sample = torch.randn(shape, device=DEVICE)
    for probability in (0.0, 0.25, 1.0):
        actual, mask = dropout_triton(sample, probability, seed, return_mask=True)
        torch.testing.assert_close(actual, dropout_torch(sample, mask, probability), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual, dropout_triton(sample, probability, seed), rtol=0, atol=0)
        torch.testing.assert_close(actual, dropout_triton(sample, probability, seed, block=512), rtol=0, atol=0)
        torch.testing.assert_close(dropout_triton(sample, probability, seed, training=False), sample, rtol=0, atol=0)
sample = torch.ones(65536, device=DEVICE)
_, mask = dropout_triton(sample, 0.25, seed, return_mask=True)
_, different_mask = dropout_triton(sample, 0.25, seed + 1, return_mask=True)
assert not torch.equal(mask, different_mask)
assert abs(mask.float().mean().item() - 0.75) < 0.02
print("Mask arithmetic, endpoint, determinism, and empirical-rate checks passed.")


# 7. Benchmark a fused RNG path and its arithmetic baseline
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

sample = torch.randn(2**20, device=DEVICE)
_, mask = dropout_triton(sample, p, seed, return_mask=True)
arithmetic_ms = bench_ms(lambda: dropout_torch(sample, mask, p))
triton_ms = bench_ms(lambda: dropout_triton(sample, p, seed))
native_ms = bench_ms(lambda: F.dropout(sample, p=p, training=True))
print(f"PyTorch supplied-mask arithmetic: {arithmetic_ms*1000:.3f} us")
print(f"PyTorch native dropout with RNG:  {native_ms*1000:.3f} us")
print(f"Triton seeded dropout with RNG:   {triton_ms*1000:.3f} us")


# 8. Register a first-order backward that regenerates the mask
from torch.autograd.function import once_differentiable


class SeededDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed, training):
        ctx.p = float(p) if training else 0.0
        ctx.seed = seed
        return dropout_triton(x, p, seed, training)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        grad_x = dropout_triton(grad_out.contiguous(), ctx.p, ctx.seed)
        return grad_x, None, None, None


for probability, training in ((0.25, True), (0.0, True), (1.0, True), (0.25, False)):
    sample = torch.randn(3, 17, device=DEVICE, requires_grad=True)
    upstream = torch.randn_like(sample)
    out = SeededDropout.apply(sample, probability, seed, training)
    out.backward(upstream)
    effective = probability if training else 0.0
    _, mask = dropout_triton(sample.detach(), effective, seed, return_mask=True)
    expected_grad = dropout_torch(upstream, mask, effective)
    torch.testing.assert_close(sample.grad, expected_grad, rtol=1e-6, atol=1e-6)
print("Regenerated first-order backward checks passed.")
```
