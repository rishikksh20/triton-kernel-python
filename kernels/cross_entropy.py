"""Per-row cross entropy: loss = logsumexp(logits) - logits[target].

Buffers: contiguous FP32 X (R,C), INT64 TARGET (R,), FP32 LOSS (R,).
Require R>=1, 1<=C<=4096, R*C < 2**31, finite logits.
Every target must be in [0,C) or equal to the ignored index -100.
Launch: cross_entropy_kernel[(R,)](X,TARGET,LOSS,C,
    triton.next_power_of_2(C),num_warps=4)
Ignored rows produce zero loss. Output is unreduced: no averaging, class
weights, label smoothing, or backward computation is performed.

Source: tutorials/25_cross_entropy.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(X, TARGET, LOSS, C: tl.constexpr, BLOCK: tl.constexpr):
    # Each program owns one example and its scalar, unreduced loss.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=-float("inf"))
    # Stable logsumexp=max+log(sum(exp(x-max))); padding contributes zero.
    maximum = tl.max(x, axis=0)
    denominator = tl.sum(tl.exp(x - maximum), axis=0)
    target = tl.load(TARGET + row)
    # Mask the target load for ignored examples: -100 is not a valid address.
    valid = target != -100
    selected = tl.load(X + row * C + target, mask=valid, other=0)
    # Grouping (maximum-selected) avoids subtracting two large logit terms later.
    loss = (maximum - selected) + tl.log(denominator)
    tl.store(LOSS + row, tl.where(valid, loss, 0.0))
