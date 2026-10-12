"""Split single-query decode into parallel cache chunks, then merge states.

Contiguous FP32 Q/OUT (B,HQ,D), KC/VC (B,HKV,C,D), LSE (B,HQ),
PM/PL (B,HQ,SPLITS), PU (B,HQ,SPLITS,D); INT32 LENGTHS (B,).
Positive dimensions, HQ%HKV=0, D in [1,128], LENGTHS in [1,C].
SPLITS in [1,128], CHUNK=ceil(C/SPLITS), BT=32, BD=next_power_of_2(D).
Finite moderate inputs; initialized valid prefixes; element counts <2**31.

Launch on one stream, num_warps=4:
  split_kv_partial_kernel[(B*HQ,SPLITS)](Q,KC,VC,LENGTHS,PM,PL,PU,
      HQ,HKV,C,D,SPLITS,triton.cdiv(C,SPLITS),32,triton.next_power_of_2(D))
  split_kv_merge_kernel[(B*HQ,)](PM,PL,PU,OUT,LSE,D,SPLITS,
      triton.next_power_of_2(SPLITS),triton.next_power_of_2(D))
All scratch entries are written on every partial launch; no initialization.
Empty chunks store m=-inf, l=0, u=0. Merge treats l=0 as inactive and
rescales unnormalized states to a common maximum before normalization.

Source: tutorials/29_split_kv_decode.md
The caller allocates/validates buffers on one CUDA GPU. Offsets are elements.
"""

import triton
import triton.language as tl


@triton.jit
def split_kv_partial_kernel(
    Q,
    KC,
    VC,
    LENGTHS,
    PM,
    PL,
    PU,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    row, split = tl.program_id(0), tl.program_id(1)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, BT)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    begin = split * CHUNK
    end = tl.minimum(begin + CHUNK, length)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    # Empty chunks skip the loop, avoiding exp(-inf - -inf).
    for start in range(begin, end, BT):
        t = start + token
        offset = ((batch * HKV + kv_head) * C + t[:, None]) * D + feature[None, :]
        valid = (t[:, None] < end) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < end, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    state = row * SPLITS + split
    tl.store(PM + state, maximum)
    tl.store(PL + state, denominator)
    tl.store(PU + state * D + feature, numerator, feature < D)


@triton.jit
def split_kv_merge_kernel(
    PM,
    PL,
    PU,
    OUT,
    LSE,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    split, feature = tl.arange(0, BS), tl.arange(0, BD)
    state = row * SPLITS + split
    maxima = tl.load(PM + state, split < SPLITS, other=-float("inf"))
    lengths = tl.load(PL + state, split < SPLITS, other=0)
    active = lengths > 0
    maximum = tl.max(tl.where(active, maxima, -float("inf")), axis=0)
    # Replace inactive maxima BEFORE exp: tl.where evaluates both operands.
    safe_maxima = tl.where(active, maxima, maximum)
    correction = tl.where(active, tl.exp(safe_maxima - maximum), 0.0)
    denominator = tl.sum(lengths * correction, axis=0)
    numerators = tl.load(
        PU + state[:, None] * D + feature[None, :],
        (split[:, None] < SPLITS) & (feature[None, :] < D),
        other=0,
    )
    numerator = tl.sum(numerators * correction[:, None], axis=0)
    # LENGTHS>=1 ensures a finite global maximum and positive denominator.
    tl.store(OUT + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))
