# 30. Fused training operations: backward kernels for SwiGLU, RoPE, and cross entropy

Previous: [Split-KV decode](29_split_kv_decode.md). Next: [Expert computation](31_expert_computation.md). Start of the series: [vector addition](01_vector_addition.md).

This self-contained Markdown notebook adds first-order derivatives to [SwiGLU](24_swiglu.md), [RoPE](20_rope.md), and [cross entropy](25_cross_entropy.md). Run sections 1–8 in order, or run section 9 independently. Floating buffers are contiguous FP32 on one NVIDIA CUDA GPU, with finite moderate values and element counts below `2**31`. SwiGLU uses saved completed gate/up projections. RoPE uses adjacent pairs, even D=2–128, and fixed cos/sin tables with valid absolute positions. Cross entropy supports C=1–4096 and ignored target -100.

The commented device code is in [fused_training_operations.py](../kernels/fused_training_operations.py). These raw kernels return gradients; the wrappers here do not register custom autograd functions. PyTorch autograd supplies an independent reference. Higher-order derivatives, gradients of RoPE tables, cross-entropy class weights, and label smoothing are outside this guide.

## Concepts: fuse work that shares loaded values

Forward fusion saves intermediate traffic, but training also needs the derivative of the fused operation. Start from the chain rule and decide which forward quantities to save or recompute.

### 1. SwiGLU: two gradients from one gate/up load

For completed projections `G=X@Wg`, `U=X@Wu`, let `Y=SiLU(G)*U`. If H is the upstream gradient of Y and `a=sigmoid(G)`, then

$$
dG=H\odot U\odot[a+G\odot a\odot(1-a)],\qquad
dU=H\odot G\odot a.
$$

One elementwise program loads G, U, and H, computes a once, and writes both gradient buffers. This fuses the activation backward, rather than the projection reductions. The remaining projection derivatives are

$$
dX=dG W_g^T+dU W_u^T,\qquad
dW_g=X^T dG,\qquad dW_u=X^T dU.
$$

Section 6 assembles these with separate PyTorch GEMMs to verify the full chain. The forward kernel in tutorial 24 writes only Y, so it cannot supply saved G/U directly. A training forward must either save those projections or recompute them during backward. Saving uses memory; recomputation adds GEMM work.

### 2. RoPE: transpose the rotation

For each adjacent pair, the forward matrix is `R=[[c,-s],[s,c]]`. With fixed c/s, the input gradient is `R.T @ dY`:

$$
dx_{even}=c\,dy_{even}+s\,dy_{odd},\qquad
dx_{odd}=-s\,dy_{even}+c\,dy_{odd}.
$$

Tables with `c=cos(theta),s=sin(theta)` make this the inverse rotation. The transpose is the derivative even if tables are not exactly orthonormal. Reuse the same absolute positions and adjacent-pair convention as forward. No original X is needed for this input gradient because the operation is linear in X.

### 3. Cross entropy: reuse the softmax reduction for dX

For an accepted target y, `loss=logsumexp(x)-x_y` and

$$
\frac{\partial L}{\partial x_c}=h\,[p_c-\mathbf{1}_{c=y}],\qquad
p_c=\frac{e^{x_c-m}}{\sum_j e^{x_j-m}}.
$$

Here h is the upstream gradient of that row's **unreduced** loss. The kernel computes a stable maximum/denominator, then writes loss and h-scaled logits gradients without storing a probability matrix. Ignored rows return zero for both, and their invalid target load is masked.

For a mean over accepted rows, h is `1/N_valid`; section 8 defines zero loss and zero gradient when every row is ignored by clamping the divisor to 1. This is an explicit convention. PyTorch's ordinary mean over no accepted rows may return NaN, so compare with a sum divided by that safe divisor.

### 4. An upstream gradient must already be known

The combined CE kernel is useful when loss and gradient are requested together with a known upstream weight, such as a sum or mean reduction. A general autograd forward does not yet know a later upstream gradient. To integrate it, save logits/targets and invoke this kernel from backward, accepting the recomputation of loss, or split the forward and backward interfaces. See [attention backward](17_flashattention_backward.md) for the surrounding custom-autograd pattern.

```text
saved G/U + dY -> fused activation derivative -> dG/dU -> projection GEMMs
dY + positions + cos/sin -> transposed pair rotation -> dX
logits + targets + row upstream -> stable reduction -> loss and dX
```

## 1. Imports and saved projection values

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
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


gate = torch.tensor([[-3., -1., 0., 1., 3.]], device=DEVICE)
up = torch.tensor([[2., -1., 3., 0.5, -2.]], device=DEVICE)
dy = torch.tensor([[1., 2., -1., 3., 0.25]], device=DEVICE)
```

Both gate signs and a zero gate matter for the SiLU derivative.

## 2. PyTorch eager: obtain two SwiGLU gradients

```python
g_ref = gate.clone().requires_grad_()
u_ref = up.clone().requires_grad_()
eager_y = F.silu(g_ref) * u_ref
expected_dg, expected_du = torch.autograd.grad(eager_y, (g_ref, u_ref), dy)
print("dGate:", expected_dg.cpu())
print("dUp:", expected_du.cpu())
```

Using arbitrary dY checks the chain rule, rather than only the derivative of a summed output.

## 3. PyTorch functions: RoPE and weighted cross entropy

```python
def rope_torch(x, positions, cos, sin):
    c = cos[positions.long()][:, None]
    s = sin[positions.long()][:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)


def cross_entropy_torch(logits, target, dloss):
    leaf = logits.detach().clone().requires_grad_()
    loss = F.cross_entropy(leaf, target, reduction="none", ignore_index=-100)
    gradient, = torch.autograd.grad(loss, leaf, dloss)
    return loss.detach(), gradient
```

RoPE builds the adjacent-pair forward independently. CE differentiates PyTorch’s unreduced loss with exactly the supplied row upstream.

## 4. Triton: two elementwise backward kernels and a CE reduction

```python
@triton.jit
def swiglu_backward_kernel(G, U, DY, DG, DU, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + index, index < SIZE, other=0)
    u = tl.load(U + index, index < SIZE, other=0)
    dy = tl.load(DY + index, index < SIZE, other=0)
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    # d SiLU(g)/dg = sigmoid(g) + g*sigmoid(g)*(1-sigmoid(g)).
    derivative = sigmoid + g * sigmoid * (1.0 - sigmoid)
    tl.store(DG + index, dy * u * derivative, index < SIZE)
    tl.store(DU + index, dy * g * sigmoid, index < SIZE)


@triton.jit
def rope_backward_kernel(
    DY,
    POS,
    COS,
    SIN,
    DX,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, token = row // (H * T), row % T
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    position = tl.load(POS + batch * T + token)
    c = tl.load(COS + position * (D // 2) + pair, valid, other=0)
    s = tl.load(SIN + position * (D // 2) + pair, valid, other=0)
    even = tl.load(DY + row * D + 2 * pair, valid, other=0)
    odd = tl.load(DY + row * D + 2 * pair + 1, valid, other=0)
    # Backpropagate with the transpose [c,s; -s,c] of the forward rotation.
    tl.store(DX + row * D + 2 * pair, even * c + odd * s, valid)
    tl.store(DX + row * D + 2 * pair + 1, -even * s + odd * c, valid)


@triton.jit
def cross_entropy_train_kernel(
    X, TARGET, DLOSS, LOSS, DX, C: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    denominator = tl.sum(exponential, axis=0)
    target = tl.load(TARGET + row)
    accepted = target != -100
    selected = tl.load(X + row * C + target, accepted, other=0)
    loss = (maximum - selected) + tl.log(denominator)
    upstream = tl.load(DLOSS + row)
    # Reuse the forward reduction to form p-one_hot(target), without storing p.
    gradient = (exponential / denominator - (col == target).to(tl.float32)) * upstream
    tl.store(LOSS + row, tl.where(accepted, loss, 0.0))
    tl.store(DX + row * C + col, tl.where(accepted, gradient, 0.0), col < C)
```

SwiGLU and RoPE require no inter-program coordination. CE owns one full class row and performs the shared reduction once.

## 5. Python wrappers: enforce each backward contract

```python
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def swiglu_backward_triton(gate, up, dy):
    require_fp32(gate, up, dy)
    if gate.shape != up.shape or gate.shape != dy.shape or gate.numel() == 0:
        raise ValueError("gate/up/dY must have matching nonempty shapes")
    dg, du = torch.empty_like(gate), torch.empty_like(up)
    swiglu_backward_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate, up, dy, dg, du, gate.numel(), 256, num_warps=4)
    return dg, du


def rope_backward_triton(dy, positions, cos, sin):
    require_fp32(dy, cos, sin)
    if dy.ndim != 4 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("require dY (B,H,T,D), matching tables (P,D/2)")
    batch, heads, tokens, d = dy.shape
    if min(batch, heads, tokens, cos.shape[0]) < 1 or d % 2 or not 2 <= d <= 128 or cos.shape[1] != d//2:
        raise ValueError("require positive dimensions and even D=2..128")
    if (positions.dtype != torch.int32 or positions.device != dy.device
        or positions.shape != (batch, tokens) or not positions.is_contiguous()):
        raise ValueError("positions must be contiguous INT32 (B,T) on the dY device")
    if bool(((positions < 0) | (positions >= cos.shape[0])).any()):
        raise ValueError("positions are outside the supplied tables")
    dx = torch.empty_like(dy)
    rope_backward_kernel[(batch*heads*tokens,)](dy, positions, cos, sin, dx,
        heads, tokens, d, triton.next_power_of_2(d//2), num_warps=4, enable_fp_fusion=False)
    return dx


def cross_entropy_train_prevalidated(logits, target, dloss):
    rows, classes = logits.shape
    loss, dx = torch.empty(rows, device=logits.device), torch.empty_like(logits)
    cross_entropy_train_kernel[(rows,)](logits, target, dloss, loss, dx,
        classes, triton.next_power_of_2(classes), num_warps=4)
    return loss, dx


def cross_entropy_train_triton(logits, target, dloss):
    require_fp32(logits, dloss)
    if logits.ndim != 2 or logits.shape[0] < 1 or not 1 <= logits.shape[1] <= 4096:
        raise ValueError("require logits (R,C), R>=1, C=1..4096")
    rows, classes = logits.shape
    if (target.dtype != torch.int64 or target.device != logits.device
        or target.shape != (rows,) or not target.is_contiguous() or dloss.shape != (rows,)):
        raise ValueError("require INT64 targets and FP32 upstream, both (R,)")
    if bool(((target != -100) & ((target < 0) | (target >= classes))).any()):
        raise ValueError("target must be a valid class or -100")
    return cross_entropy_train_prevalidated(logits, target, dloss)


dg, du = swiglu_backward_triton(gate, up, dy)
torch.testing.assert_close(dg, expected_dg, rtol=2e-5, atol=2e-5)
torch.testing.assert_close(du, expected_du, rtol=2e-5, atol=2e-5)
```

The supplied dY and row upstream are inputs to differentiation. They can be negative, zero, or nonuniform; treating every upstream as 1 would be wrong.

## 6. Verify full projection gradients, RoPE positions, and ignored CE rows

```python
# Full SwiGLU chain, with GEMMs outside the activation-backward kernel.
x = torch.randn(17, 13, device=DEVICE, requires_grad=True)
wg = (torch.randn(13, 19, device=DEVICE) / 13**0.5).requires_grad_()
wu = (torch.randn(13, 19, device=DEVICE) / 13**0.5).requires_grad_()
full_g, full_u = x @ wg, x @ wu
full_dy = torch.randn(17, 19, device=DEVICE)
expected = torch.autograd.grad(F.silu(full_g)*full_u, (x, wg, wu), full_dy)
dg, du = swiglu_backward_triton(full_g.detach(), full_u.detach(), full_dy)
actual = (dg @ wg.detach().T + du @ wu.detach().T, x.detach().T @ dg, x.detach().T @ du)
for got, want in zip(actual, expected):
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
for d in (2, 14, 128):
    sample = torch.randn(2, 3, 5, d, device=DEVICE, requires_grad=True)
    theta = torch.randn(11, d//2, device=DEVICE)
    cos, sin = theta.cos(), theta.sin()
    positions = torch.tensor([[0, 1, 3, 7, 10], [2, 2, 4, 6, 8]], device=DEVICE, dtype=torch.int32)
    upstream = torch.randn_like(sample)
    expected_dx, = torch.autograd.grad(rope_torch(sample, positions, cos, sin), sample, upstream)
    actual_dx = rope_backward_triton(upstream, positions, cos, sin)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-5, atol=2e-5)
for classes in (1, 7, 33, 257):
    logits = torch.randn(5, classes, device=DEVICE)
    target = torch.tensor([0, classes-1, -100, 0, -100], device=DEVICE)
    row_upstream = torch.tensor([1., -2., 3., 0., 0.5], device=DEVICE)
    loss, gradient = cross_entropy_train_triton(logits, target, row_upstream)
    expected_loss, expected_gradient = cross_entropy_torch(logits, target, row_upstream)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradient.sum(-1), torch.zeros(5, device=DEVICE), rtol=0, atol=2e-6)
    assert torch.count_nonzero(gradient[target == -100]).item() == 0
print("SwiGLU projection chain, RoPE backward, and CE loss/gradient checks passed.")
```

The full SwiGLU chain has uneven dimensions. RoPE repeats and varies absolute positions. CE includes C=1, a class-axis tail, ignored rows, and arbitrary row gradients; each accepted gradient row sums approximately to zero.

## 7. Benchmark CE loss plus its weighted logits gradient

```python
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_logits = torch.randn(128, 1024, device=DEVICE)
timed_target = torch.arange(128, device=DEVICE, dtype=torch.int64) % 1024
timed_upstream = torch.full((128,), 1/128, device=DEVICE)
loss, gradient = cross_entropy_train_triton(timed_logits, timed_target, timed_upstream)
expected_loss, expected_gradient = cross_entropy_torch(timed_logits, timed_target, timed_upstream)
torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-5)
torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
p = bench_ms(lambda: cross_entropy_torch(timed_logits, timed_target, timed_upstream))
t = bench_ms(lambda: cross_entropy_train_prevalidated(timed_logits, timed_target, timed_upstream))
print(f"CE loss + gradient: PyTorch {p*1000:.3f} us; Triton {t*1000:.3f} us")
```

Both paths return unreduced loss and a logits-gradient buffer. The PyTorch reference also clones logits to create an autograd leaf, so this measures these educational functions, rather than matched production training pipelines. GPU event timing does not include all Python autograd overhead.

## 8. Supply a safe mean-reduction upstream, including all-ignored input

```python
for target_values in ([0, -100, 2], [-100, -100, -100]):
    logits = torch.randn(3, 7, device=DEVICE)
    target = torch.tensor(target_values, device=DEVICE, dtype=torch.int64)
    count = (target != -100).sum().clamp_min(1)
    dloss = torch.ones(3, device=DEVICE) / count
    losses, dx = cross_entropy_train_triton(logits, target, dloss)
    mean_loss = losses.sum() / count
    leaf = logits.clone().requires_grad_()
    safe_mean = F.cross_entropy(leaf, target, reduction="sum", ignore_index=-100) / count
    expected_dx, = torch.autograd.grad(safe_mean, leaf)
    torch.testing.assert_close(mean_loss, safe_mean.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(dx, expected_dx, rtol=2e-5, atol=2e-5)
    print("Accepted rows:", int((target != -100).sum()), "safe mean:", mean_loss.item())
```

A custom autograd integration must multiply by any later scalar upstream as well. Raw returned gradient buffers do not themselves make a differentiable PyTorch operation. Test the same reduction, ignored-row convention, and saved/recomputed state when adding that integration.

## Try next

Continue with [expert computation](31_expert_computation.md). Extend SwiGLU backward with your own projection GEMMs, or use the custom-autograd structure from tutorial 17 while keeping first-order gradient checks against PyTorch.

## 9. Complete code in one block

Run this block independently in a fresh notebook, or save it as a Python script. It combines sections 1–8; the kernel definitions match the linked Triton-only file.

```python
# 1. Imports and saved projection values
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.testing

if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("This notebook requires a working NVIDIA CUDA GPU.")
DEVICE = torch.device("cuda", torch.cuda.current_device())
torch.manual_seed(0)
torch.set_float32_matmul_precision("highest")
print("PyTorch:", torch.__version__, "Triton:", triton.__version__)
print("GPU:", torch.cuda.get_device_name(DEVICE))


gate = torch.tensor([[-3., -1., 0., 1., 3.]], device=DEVICE)
up = torch.tensor([[2., -1., 3., 0.5, -2.]], device=DEVICE)
dy = torch.tensor([[1., 2., -1., 3., 0.25]], device=DEVICE)


# 2. PyTorch eager: obtain two SwiGLU gradients
g_ref = gate.clone().requires_grad_()
u_ref = up.clone().requires_grad_()
eager_y = F.silu(g_ref) * u_ref
expected_dg, expected_du = torch.autograd.grad(eager_y, (g_ref, u_ref), dy)
print("dGate:", expected_dg.cpu())
print("dUp:", expected_du.cpu())


# 3. PyTorch functions: RoPE and weighted cross entropy
def rope_torch(x, positions, cos, sin):
    c = cos[positions.long()][:, None]
    s = sin[positions.long()][:, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even*c - odd*s, even*s + odd*c), dim=-1).flatten(-2)


def cross_entropy_torch(logits, target, dloss):
    leaf = logits.detach().clone().requires_grad_()
    loss = F.cross_entropy(leaf, target, reduction="none", ignore_index=-100)
    gradient, = torch.autograd.grad(loss, leaf, dloss)
    return loss.detach(), gradient


# 4. Triton: two elementwise backward kernels and a CE reduction
@triton.jit
def swiglu_backward_kernel(G, U, DY, DG, DU, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(G + index, index < SIZE, other=0)
    u = tl.load(U + index, index < SIZE, other=0)
    dy = tl.load(DY + index, index < SIZE, other=0)
    sigmoid = 1.0 / (1.0 + tl.exp(-g))
    # d SiLU(g)/dg = sigmoid(g) + g*sigmoid(g)*(1-sigmoid(g)).
    derivative = sigmoid + g * sigmoid * (1.0 - sigmoid)
    tl.store(DG + index, dy * u * derivative, index < SIZE)
    tl.store(DU + index, dy * g * sigmoid, index < SIZE)


@triton.jit
def rope_backward_kernel(
    DY,
    POS,
    COS,
    SIN,
    DX,
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    batch, token = row // (H * T), row % T
    pair = tl.arange(0, BLOCK)
    valid = pair < D // 2
    position = tl.load(POS + batch * T + token)
    c = tl.load(COS + position * (D // 2) + pair, valid, other=0)
    s = tl.load(SIN + position * (D // 2) + pair, valid, other=0)
    even = tl.load(DY + row * D + 2 * pair, valid, other=0)
    odd = tl.load(DY + row * D + 2 * pair + 1, valid, other=0)
    # Backpropagate with the transpose [c,s; -s,c] of the forward rotation.
    tl.store(DX + row * D + 2 * pair, even * c + odd * s, valid)
    tl.store(DX + row * D + 2 * pair + 1, -even * s + odd * c, valid)


@triton.jit
def cross_entropy_train_kernel(
    X, TARGET, DLOSS, LOSS, DX, C: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=-float("inf"))
    maximum = tl.max(x, axis=0)
    exponential = tl.exp(x - maximum)
    denominator = tl.sum(exponential, axis=0)
    target = tl.load(TARGET + row)
    accepted = target != -100
    selected = tl.load(X + row * C + target, accepted, other=0)
    loss = (maximum - selected) + tl.log(denominator)
    upstream = tl.load(DLOSS + row)
    # Reuse the forward reduction to form p-one_hot(target), without storing p.
    gradient = (exponential / denominator - (col == target).to(tl.float32)) * upstream
    tl.store(LOSS + row, tl.where(accepted, loss, 0.0))
    tl.store(DX + row * C + col, tl.where(accepted, gradient, 0.0), col < C)


# 5. Python wrappers: enforce each backward contract
def require_fp32(*tensors):
    first = tensors[0]
    if not first.is_cuda or any(
        t.dtype != torch.float32 or t.device != first.device
        or not t.is_contiguous() or t.numel() >= 2**31 for t in tensors
    ):
        raise ValueError("require contiguous FP32 on one CUDA GPU, offsets < 2**31")



def swiglu_backward_triton(gate, up, dy):
    require_fp32(gate, up, dy)
    if gate.shape != up.shape or gate.shape != dy.shape or gate.numel() == 0:
        raise ValueError("gate/up/dY must have matching nonempty shapes")
    dg, du = torch.empty_like(gate), torch.empty_like(up)
    swiglu_backward_kernel[(triton.cdiv(gate.numel(), 256),)](
        gate, up, dy, dg, du, gate.numel(), 256, num_warps=4)
    return dg, du


def rope_backward_triton(dy, positions, cos, sin):
    require_fp32(dy, cos, sin)
    if dy.ndim != 4 or cos.ndim != 2 or sin.shape != cos.shape:
        raise ValueError("require dY (B,H,T,D), matching tables (P,D/2)")
    batch, heads, tokens, d = dy.shape
    if min(batch, heads, tokens, cos.shape[0]) < 1 or d % 2 or not 2 <= d <= 128 or cos.shape[1] != d//2:
        raise ValueError("require positive dimensions and even D=2..128")
    if (positions.dtype != torch.int32 or positions.device != dy.device
        or positions.shape != (batch, tokens) or not positions.is_contiguous()):
        raise ValueError("positions must be contiguous INT32 (B,T) on the dY device")
    if bool(((positions < 0) | (positions >= cos.shape[0])).any()):
        raise ValueError("positions are outside the supplied tables")
    dx = torch.empty_like(dy)
    rope_backward_kernel[(batch*heads*tokens,)](dy, positions, cos, sin, dx,
        heads, tokens, d, triton.next_power_of_2(d//2), num_warps=4, enable_fp_fusion=False)
    return dx


def cross_entropy_train_prevalidated(logits, target, dloss):
    rows, classes = logits.shape
    loss, dx = torch.empty(rows, device=logits.device), torch.empty_like(logits)
    cross_entropy_train_kernel[(rows,)](logits, target, dloss, loss, dx,
        classes, triton.next_power_of_2(classes), num_warps=4)
    return loss, dx


def cross_entropy_train_triton(logits, target, dloss):
    require_fp32(logits, dloss)
    if logits.ndim != 2 or logits.shape[0] < 1 or not 1 <= logits.shape[1] <= 4096:
        raise ValueError("require logits (R,C), R>=1, C=1..4096")
    rows, classes = logits.shape
    if (target.dtype != torch.int64 or target.device != logits.device
        or target.shape != (rows,) or not target.is_contiguous() or dloss.shape != (rows,)):
        raise ValueError("require INT64 targets and FP32 upstream, both (R,)")
    if bool(((target != -100) & ((target < 0) | (target >= classes))).any()):
        raise ValueError("target must be a valid class or -100")
    return cross_entropy_train_prevalidated(logits, target, dloss)


dg, du = swiglu_backward_triton(gate, up, dy)
torch.testing.assert_close(dg, expected_dg, rtol=2e-5, atol=2e-5)
torch.testing.assert_close(du, expected_du, rtol=2e-5, atol=2e-5)


# 6. Verify full projection gradients, RoPE positions, and ignored CE rows
# Full SwiGLU chain, with GEMMs outside the activation-backward kernel.
x = torch.randn(17, 13, device=DEVICE, requires_grad=True)
wg = (torch.randn(13, 19, device=DEVICE) / 13**0.5).requires_grad_()
wu = (torch.randn(13, 19, device=DEVICE) / 13**0.5).requires_grad_()
full_g, full_u = x @ wg, x @ wu
full_dy = torch.randn(17, 19, device=DEVICE)
expected = torch.autograd.grad(F.silu(full_g)*full_u, (x, wg, wu), full_dy)
dg, du = swiglu_backward_triton(full_g.detach(), full_u.detach(), full_dy)
actual = (dg @ wg.detach().T + du @ wu.detach().T, x.detach().T @ dg, x.detach().T @ du)
for got, want in zip(actual, expected):
    torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-4)
for d in (2, 14, 128):
    sample = torch.randn(2, 3, 5, d, device=DEVICE, requires_grad=True)
    theta = torch.randn(11, d//2, device=DEVICE)
    cos, sin = theta.cos(), theta.sin()
    positions = torch.tensor([[0, 1, 3, 7, 10], [2, 2, 4, 6, 8]], device=DEVICE, dtype=torch.int32)
    upstream = torch.randn_like(sample)
    expected_dx, = torch.autograd.grad(rope_torch(sample, positions, cos, sin), sample, upstream)
    actual_dx = rope_backward_triton(upstream, positions, cos, sin)
    torch.testing.assert_close(actual_dx, expected_dx, rtol=2e-5, atol=2e-5)
for classes in (1, 7, 33, 257):
    logits = torch.randn(5, classes, device=DEVICE)
    target = torch.tensor([0, classes-1, -100, 0, -100], device=DEVICE)
    row_upstream = torch.tensor([1., -2., 3., 0., 0.5], device=DEVICE)
    loss, gradient = cross_entropy_train_triton(logits, target, row_upstream)
    expected_loss, expected_gradient = cross_entropy_torch(logits, target, row_upstream)
    torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(gradient.sum(-1), torch.zeros(5, device=DEVICE), rtol=0, atol=2e-6)
    assert torch.count_nonzero(gradient[target == -100]).item() == 0
print("SwiGLU projection chain, RoPE backward, and CE loss/gradient checks passed.")


# 7. Benchmark CE loss plus its weighted logits gradient
def bench_ms(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize(DEVICE)
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


timed_logits = torch.randn(128, 1024, device=DEVICE)
timed_target = torch.arange(128, device=DEVICE, dtype=torch.int64) % 1024
timed_upstream = torch.full((128,), 1/128, device=DEVICE)
loss, gradient = cross_entropy_train_triton(timed_logits, timed_target, timed_upstream)
expected_loss, expected_gradient = cross_entropy_torch(timed_logits, timed_target, timed_upstream)
torch.testing.assert_close(loss, expected_loss, rtol=2e-5, atol=2e-5)
torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=2e-5)
p = bench_ms(lambda: cross_entropy_torch(timed_logits, timed_target, timed_upstream))
t = bench_ms(lambda: cross_entropy_train_prevalidated(timed_logits, timed_target, timed_upstream))
print(f"CE loss + gradient: PyTorch {p*1000:.3f} us; Triton {t*1000:.3f} us")


# 8. Supply a safe mean-reduction upstream, including all-ignored input
for target_values in ([0, -100, 2], [-100, -100, -100]):
    logits = torch.randn(3, 7, device=DEVICE)
    target = torch.tensor(target_values, device=DEVICE, dtype=torch.int64)
    count = (target != -100).sum().clamp_min(1)
    dloss = torch.ones(3, device=DEVICE) / count
    losses, dx = cross_entropy_train_triton(logits, target, dloss)
    mean_loss = losses.sum() / count
    leaf = logits.clone().requires_grad_()
    safe_mean = F.cross_entropy(leaf, target, reduction="sum", ignore_index=-100) / count
    expected_dx, = torch.autograd.grad(safe_mean, leaf)
    torch.testing.assert_close(mean_loss, safe_mean.detach(), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(dx, expected_dx, rtol=2e-5, atol=2e-5)
    print("Accepted rows:", int((target != -100).sum()), "safe mean:", mean_loss.item())
```
