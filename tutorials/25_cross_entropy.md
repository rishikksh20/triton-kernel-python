# 25. Cross entropy: fuse log-sum-exp with target indexing

Previous: [SwiGLU](24_swiglu.md). Next: [MoE routing](26_moe_routing.md). Start of the series: [vector addition](01_vector_addition.md).

This is a self-contained Markdown notebook. Run sections 1–8 in order, or run the final complete code block independently. Logits are finite, moderate, contiguous FP32 `(R,C)` tensors on one NVIDIA CUDA GPU with R >= 1 and C=1–4096. Targets are contiguous INT64 `(R,)`, containing class indices in `[0,C)` or ignore_index=-100. The main Triton operation returns unreduced per-row losses and has no registered PyTorch backward operation. There are no class weights, soft targets, or label smoothing.

## Concepts: calculate the negative log probability without storing softmax

For row r and target class t_r, cross entropy is

$$
L_r=-\log\operatorname{softmax}(x_r)_{t_r}
=\log\sum_c e^{x_{r,c}}-x_{r,t_r}.
$$

Shift by the row maximum m_r to obtain the stable form used here:

$$
L_r=(m_r-x_{r,t_r})+\log\sum_c e^{x_{r,c}-m_r}.
$$

### 1. Reduce across classes and gather one target

One program loads a row, reduces its maximum, reduces the shifted exponential sum, and loads the selected logit through a scalar target pointer. It stores one FP32 loss. No full softmax or log-softmax tensor is written. Padded classes use negative infinity for the maximum and exponential reduction.

### 2. Ignored rows contribute zero

When a target is -100, return zero and avoid indexing logits with that negative target. The class mask and the ignored-row mask do different jobs: padding removes nonexistent classes, while ignore_index removes an entire row from the loss. The [PyTorch cross-entropy API](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) describes the target and reduction contracts.

### 3. Reduction determines the denominator

`sum` adds per-row losses. For unweighted integer targets, `mean` divides by the number of nonignored rows, not R. An all-ignored mean is undefined and PyTorch returns NaN; an all-ignored sum is zero. The main kernel computes `reduction='none'`; section 8 performs sum/mean in PyTorch and discusses the gradient separately.

```text
logits row -> maximum -> shifted exp sum --+
target index -> selected logit ------------+ -> one loss
ignore_index -> zero row loss
```

Adding the same constant to every class in a row preserves softmax and the loss in real arithmetic. Testing that invariance catches unshifted exponentiation and checks stability. The row fits in one program; large vocabularies need a different multi-stage or streaming schedule.

## 1. Imports and targets including an ignored row

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

logits = torch.tensor([[2., 0., -1.], [1., 1., 1.], [-3., 4., 0.]], device=DEVICE)
targets = torch.tensor([0, 2, -100], device=DEVICE, dtype=torch.int64)
```

The uniform second row has loss log(3), and the ignored third row has zero loss.

## 2. PyTorch eager: select from stable log probabilities

```python
log_probs = logits.log_softmax(dim=-1)
safe_targets = targets.clamp_min(0)
selected = log_probs.gather(1, safe_targets[:, None]).squeeze(1)
loss_eager = torch.where(targets != -100, -selected, 0.)
print("Per-row losses:", loss_eager.cpu())
```

Safe indexing happens before the ignored-row selection. Negative ignore indices must not be used as class pointers.

## 3. PyTorch function: a public cross-entropy reference

```python
def cross_entropy_torch(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")


torch.testing.assert_close(cross_entropy_torch(logits, targets), loss_eager)
```

The reference uses logits directly; applying softmax before cross entropy would define a different computation.

## 4. Triton: stable row reduction and a guarded target load

```python
@triton.jit
def cross_entropy_kernel(X, TARGET, LOSS, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    denominator = tl.sum(tl.exp(x - maximum), axis=0)
    target = tl.load(TARGET + row)
    valid = target != -100
    selected = tl.load(X + row * C + target, mask=valid, other=0)
    loss = (maximum - selected) + tl.log(denominator)
    tl.store(LOSS + row, tl.where(valid, loss, 0.))
```

Computing `(maximum-selected)+log(denominator)` avoids subtracting two large separately assembled log-sum-exp values for a large shared shift.

## 5. Python wrapper: validate class indices before launching

```python
def cross_entropy_into(logits, targets, losses):
    rows, classes = logits.shape
    cross_entropy_kernel[(rows,)](logits, targets, losses, classes,
                                  triton.next_power_of_2(classes), num_warps=4)


def cross_entropy_triton(logits, targets):
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("logits must be FP32 (R,C)")
    rows, classes = logits.shape
    if rows < 1 or not 1 <= classes <= 4096 or logits.numel() >= 2**31:
        raise ValueError("require R >= 1, C=1..4096, and 32-bit-safe offsets")
    if targets.shape != (rows,) or targets.dtype != torch.int64:
        raise ValueError("targets must be INT64 (R,)")
    if not logits.is_cuda or targets.device != logits.device or not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("inputs must be contiguous on the same CUDA device")
    bad = (targets != -100) & ((targets < 0) | (targets >= classes))
    if bad.any().item():
        raise ValueError("target is neither a valid class index nor ignore_index=-100")
    losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
    cross_entropy_into(logits, targets, losses)
    return losses
```

Index validation synchronizes the GPU. The preallocated launch helper assumes valid target values; benchmarks validate once outside timing.

## 6. Compare loss values, class tails, and common shifts

```python
torch.testing.assert_close(cross_entropy_triton(logits, targets), loss_eager, rtol=1e-5, atol=1e-5)
for rows, classes in ((1, 1), (7, 3), (33, 127), (8, 1025), (2, 4096)):
    x = torch.randn(rows, classes, device=DEVICE)
    t = torch.randint(classes, (rows,), device=DEVICE)
    if rows > 1:
        t[0] = -100
    actual = cross_entropy_triton(x, t)
    expected = cross_entropy_torch(x, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    shifted = x + 1000.
    shifted_actual = cross_entropy_triton(shifted, t)
    torch.testing.assert_close(shifted_actual, cross_entropy_torch(shifted, t), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(shifted_actual, actual, rtol=1e-4, atol=2e-4)
print("Per-row losses, ignored rows, class tails, and shifted-logit checks passed.")
```

A large FP32 shift can round the inputs before the kernel sees them. The invariance check therefore has a wider tolerance than the comparison against equally shifted reference inputs.

## 7. Benchmark validated preallocated per-row loss

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for rows, classes in ((128, 128), (256, 1024), (128, 4096)):
    x = torch.randn(rows, classes, device=DEVICE)
    t = torch.randint(classes, (rows,), device=DEVICE)
    losses = cross_entropy_triton(x, t)
    torch.testing.assert_close(losses, cross_entropy_torch(x, t), rtol=1e-5, atol=1e-5)
    p = bench_ms(lambda: cross_entropy_torch(x, t))
    timing = bench_ms(lambda: cross_entropy_into(x, t, losses))
    print(f"{rows,classes}: PyTorch allocating {p*1000:.3f} us, Triton preallocated {timing*1000:.3f} us")
```

Both paths compute unreduced losses; neither includes mean/sum reduction. Target validation, input construction, compilation, checks, and printing are outside timing. Output allocation scopes differ as labeled.

## 8. Check reductions and the analytic first derivative

```python
losses = cross_entropy_triton(logits, targets)
valid = targets != -100
summed = losses.sum()
mean = summed / valid.sum()
torch.testing.assert_close(summed, F.cross_entropy(logits, targets, reduction="sum"))
torch.testing.assert_close(mean, F.cross_entropy(logits, targets, reduction="mean"))
all_ignored = torch.full_like(targets, -100)
ignored_losses = cross_entropy_triton(logits, all_ignored)
assert ignored_losses.sum().item() == 0.
assert torch.isnan(ignored_losses.sum() / (all_ignored != -100).sum()).item()

leaf = logits.detach().clone().requires_grad_(True)
F.cross_entropy(leaf, targets, reduction="sum").backward()
analytic = logits.softmax(-1)
analytic[valid, targets[valid]] -= 1
analytic[~valid] = 0
torch.testing.assert_close(analytic, leaf.grad, rtol=1e-5, atol=1e-5)
print("Sum/mean, all-ignored behavior, and analytic gradient checks passed.")
```

The analytic derivative is `softmax(x)-one_hot(target)` for each valid row; a mean adds the valid-row denominator. This is a PyTorch derivation check. The Triton loss still needs an explicit backward kernel/autograd registration before training can use it.

## Try next

Continue with [MoE routing](26_moe_routing.md). Extend cross entropy with a saved row statistic and a tiled gradient kernel; retain ignored-row semantics and the chosen reduction denominator.

## 9. Complete code in one block

Run this block independently in a fresh notebook cell, or save it as a Python script. It combines all code from sections 1–8; comments mark each step.

```python
# 1. Imports and targets including an ignored row
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

logits = torch.tensor([[2., 0., -1.], [1., 1., 1.], [-3., 4., 0.]], device=DEVICE)
targets = torch.tensor([0, 2, -100], device=DEVICE, dtype=torch.int64)


# 2. PyTorch eager: select from stable log probabilities
log_probs = logits.log_softmax(dim=-1)
safe_targets = targets.clamp_min(0)
selected = log_probs.gather(1, safe_targets[:, None]).squeeze(1)
loss_eager = torch.where(targets != -100, -selected, 0.)
print("Per-row losses:", loss_eager.cpu())


# 3. PyTorch function: a public cross-entropy reference
def cross_entropy_torch(logits, targets):
    return F.cross_entropy(logits, targets, ignore_index=-100, reduction="none")


torch.testing.assert_close(cross_entropy_torch(logits, targets), loss_eager)


# 4. Triton: stable row reduction and a guarded target load
@triton.jit
def cross_entropy_kernel(X, TARGET, LOSS, C: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    denominator = tl.sum(tl.exp(x - maximum), axis=0)
    target = tl.load(TARGET + row)
    valid = target != -100
    selected = tl.load(X + row * C + target, mask=valid, other=0)
    loss = (maximum - selected) + tl.log(denominator)
    tl.store(LOSS + row, tl.where(valid, loss, 0.))


# 5. Python wrapper: validate class indices before launching
def cross_entropy_into(logits, targets, losses):
    rows, classes = logits.shape
    cross_entropy_kernel[(rows,)](logits, targets, losses, classes,
                                  triton.next_power_of_2(classes), num_warps=4)


def cross_entropy_triton(logits, targets):
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("logits must be FP32 (R,C)")
    rows, classes = logits.shape
    if rows < 1 or not 1 <= classes <= 4096 or logits.numel() >= 2**31:
        raise ValueError("require R >= 1, C=1..4096, and 32-bit-safe offsets")
    if targets.shape != (rows,) or targets.dtype != torch.int64:
        raise ValueError("targets must be INT64 (R,)")
    if not logits.is_cuda or targets.device != logits.device or not logits.is_contiguous() or not targets.is_contiguous():
        raise ValueError("inputs must be contiguous on the same CUDA device")
    bad = (targets != -100) & ((targets < 0) | (targets >= classes))
    if bad.any().item():
        raise ValueError("target is neither a valid class index nor ignore_index=-100")
    losses = torch.empty(rows, device=logits.device, dtype=torch.float32)
    cross_entropy_into(logits, targets, losses)
    return losses


# 6. Compare loss values, class tails, and common shifts
torch.testing.assert_close(cross_entropy_triton(logits, targets), loss_eager, rtol=1e-5, atol=1e-5)
for rows, classes in ((1, 1), (7, 3), (33, 127), (8, 1025), (2, 4096)):
    x = torch.randn(rows, classes, device=DEVICE)
    t = torch.randint(classes, (rows,), device=DEVICE)
    if rows > 1:
        t[0] = -100
    actual = cross_entropy_triton(x, t)
    expected = cross_entropy_torch(x, t)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    shifted = x + 1000.
    shifted_actual = cross_entropy_triton(shifted, t)
    torch.testing.assert_close(shifted_actual, cross_entropy_torch(shifted, t), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(shifted_actual, actual, rtol=1e-4, atol=2e-4)
print("Per-row losses, ignored rows, class tails, and shifted-logit checks passed.")


# 7. Benchmark validated preallocated per-row loss
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")

for rows, classes in ((128, 128), (256, 1024), (128, 4096)):
    x = torch.randn(rows, classes, device=DEVICE)
    t = torch.randint(classes, (rows,), device=DEVICE)
    losses = cross_entropy_triton(x, t)
    torch.testing.assert_close(losses, cross_entropy_torch(x, t), rtol=1e-5, atol=1e-5)
    p = bench_ms(lambda: cross_entropy_torch(x, t))
    timing = bench_ms(lambda: cross_entropy_into(x, t, losses))
    print(f"{rows,classes}: PyTorch allocating {p*1000:.3f} us, Triton preallocated {timing*1000:.3f} us")


# 8. Check reductions and the analytic first derivative
losses = cross_entropy_triton(logits, targets)
valid = targets != -100
summed = losses.sum()
mean = summed / valid.sum()
torch.testing.assert_close(summed, F.cross_entropy(logits, targets, reduction="sum"))
torch.testing.assert_close(mean, F.cross_entropy(logits, targets, reduction="mean"))
all_ignored = torch.full_like(targets, -100)
ignored_losses = cross_entropy_triton(logits, all_ignored)
assert ignored_losses.sum().item() == 0.
assert torch.isnan(ignored_losses.sum() / (all_ignored != -100).sum()).item()

leaf = logits.detach().clone().requires_grad_(True)
F.cross_entropy(leaf, targets, reduction="sum").backward()
analytic = logits.softmax(-1)
analytic[valid, targets[valid]] -= 1
analytic[~valid] = 0
torch.testing.assert_close(analytic, leaf.grad, rtol=1e-5, atol=1e-5)
print("Sum/mean, all-ignored behavior, and analytic gradient checks passed.")
```
