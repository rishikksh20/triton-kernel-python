"""First-order backward kernels for SwiGLU, fixed-table RoPE, and CE.

All floating buffers are contiguous FP32 with finite moderate inputs.
SwiGLU: G/U/DY/DG/DU same shape, flattened SIZE, positive SIZE<2**31.
G and U are saved completed projections; only activation backward is fused.
Launch swiglu_backward_kernel[(triton.cdiv(SIZE,256),)](...,SIZE,256).

RoPE: DY/DX (B,H,T,D), INT32 POS (B,T), COS/SIN (P,D/2).
D is even in [2,128], 0<=POS<P; tables are fixed, not differentiable.
Launch rope_backward_kernel[(B*H*T,)](...,H,T,D,
    triton.next_power_of_2(D//2),num_warps=4,enable_fp_fusion=False).

Cross entropy: X/DX (R,C), INT64 TARGET (R,), LOSS/DLOSS (R,),
1<=C<=4096, R>=1, targets in [0,C) or -100. DLOSS is the upstream
gradient of EACH unreduced loss; a mean reduction supplies 1/N_valid.
Launch cross_entropy_train_kernel[(R,)](...,C,
    triton.next_power_of_2(C),num_warps=4).
Writes both unreduced forward loss and scaled dX, zero for ignored rows.
All element counts <2**31. No allocation, autograd registration, higher-order
gradients, class weights, label smoothing, or trigonometric table gradients.

Source: tutorials/30_fused_training_operations.md
The caller validates/allocates buffers on one CUDA GPU. Offsets are elements.
"""

import triton
import triton.language as tl


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
