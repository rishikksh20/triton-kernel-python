"""Single-query decode attention over ragged KV caches, including grouped heads.

Buffers: contiguous FP32 Q/Out (B,HQ,D), KC/VC (B,HKV,C,D), LSE (B,HQ),
INT32 LENGTHS (B,). Require B,HQ,HKV,C>0, HQ % HKV == 0, 1<=D<=128.
Valid LENGTHS entries lie in [1,C]; all slots below each length must be
initialized. Q/cache element counts must be < 2**31.
Launch: decode_kernel[(B*HQ,)](Q,KC,VC,LENGTHS,Out,LSE,HQ,HKV,C,D,
    32,triton.next_power_of_2(D),num_warps=4)
HQ=HKV gives ordinary multi-head attention; HQ>HKV shares each KV head
across a group of query heads. Each program streams one query's valid keys.
The output is normalized online; LSE uses natural logarithms.

Source: tutorials/22_decode_attention.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def decode_kernel(
    Q,
    KC,
    VC,
    LENGTHS,
    Out,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # One program handles the single query for one (batch,query_head).
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    # Consecutive HQ/HKV query heads share one KV head (grouped-query attention).
    kv_head = head // (HQ // HKV)
    cache_row = batch * HKV + kv_head
    feature = tl.arange(0, BD)
    token = tl.arange(0, BT)
    query = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    # Stream state: score maximum, shifted denominator, and value numerator.
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    # LENGTHS is runtime metadata; read only initialized, valid cache tokens.
    for start in range(0, length, BT):
        t = start + token
        mask = (t[:, None] < length) & (feature[None, :] < D)
        offsets = (cache_row * C + t[:, None]) * D + feature[None, :]
        key = tl.load(KC + offsets, mask, other=0)
        value = tl.load(VC + offsets, mask, other=0)
        # Reduce over features to obtain one scaled dot product per cache token.
        scores = tl.sum(key * query[None, :], axis=1) * (D**-0.5)
        scores = tl.where(t < length, scores, -float("inf"))
        updated = tl.maximum(maximum, tl.max(scores, axis=0))
        # Changing the maximum rescales both previously accumulated quantities.
        correction = tl.exp(maximum - updated)
        weights = tl.exp(scores - updated)
        numerator = numerator * correction + tl.sum(weights[:, None] * value, axis=0)
        denominator = denominator * correction + tl.sum(weights, axis=0)
        maximum = updated
    # Normalize once after the complete cache prefix has contributed.
    tl.store(Out + row * D + feature, numerator / denominator, feature < D)
    tl.store(LSE + row, maximum + tl.log(denominator))
