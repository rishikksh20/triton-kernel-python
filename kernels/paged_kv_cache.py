"""Paged KV allocation, single-token append, and single-query attention.

FP32 KC/VC (NPAGES,HKV,P,D), NK/NV (B,HKV,D), Q/OUT (B,HQ,D),
LSE (B,HQ); INT32 TABLE (B,MAX_PAGES), LENGTHS/OK (B,), FREE (NFREE,),
CURSOR (1,). HQ is a positive multiple of HKV; 1<=D<=128; P is a positive
power of two <=32. Other dimensions positive, NFREE>=0; element counts <2**31.
Lengths before append lie in [0,MAX_PAGES*P]. Valid prefixes are initialized.
TABLE is -1 for unallocated pages; allocated IDs are in [0,NPAGES).
Writable pages have exclusive sequence ownership. FREE contains distinct
unused physical page IDs; CURSOR starts at zero and only increases.
No freeing, eviction, prefix sharing, or copy-on-write is implemented.

Launch IN ORDER on one stream (all num_warps=4):
  allocate_pages_kernel[(B,)](TABLE,LENGTHS,FREE,CURSOR,OK,MAX_PAGES,P,NFREE)
  paged_append_kernel[(B*HKV,)](KC,VC,NK,NV,TABLE,LENGTHS,OK,
      HKV,MAX_PAGES,P,D,triton.next_power_of_2(D))
  advance_lengths_kernel[(triton.cdiv(B,128),)](LENGTHS,OK,B,128)
  paged_decode_kernel[(B*HQ,)](Q,KC,VC,TABLE,LENGTHS,OUT,LSE,
      HQ,HKV,MAX_PAGES,P,D,triton.next_power_of_2(D))
Readers run after append completes. Allocation exhaustion or logical capacity
sets OK=0 and preserves that sequence's length/cache. Failed reservations
consume cursor tickets; stop/rebuild the pool rather than retry indefinitely.
An empty sequence returns OUT=0, LSE=-inf, an explicit inference convention.

Source: tutorials/28_paged_kv_caches.md
The caller validates/allocates buffers on one CUDA GPU. Offsets are elements.
"""

import triton
import triton.language as tl


@triton.jit
def allocate_pages_kernel(
    TABLE,
    LENGTHS,
    FREE,
    CURSOR,
    OK,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    NFREE: tl.constexpr,
):
    batch = tl.program_id(0)
    position = tl.load(LENGTHS + batch)
    accepted = tl.full((), 0, tl.int32)
    if position < MAX_PAGES * P:
        logical = position // P
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        # Allocate only at page boundaries and only for an unassigned entry.
        if (position % P == 0) & (page < 0):
            ticket = tl.atomic_add(CURSOR, 1)
            page = tl.load(FREE + ticket, ticket < NFREE, other=-1)
            tl.store(TABLE + batch * MAX_PAGES + logical, page)
        accepted = (page >= 0).to(tl.int32)
    # Every batch item gets a status, even when no new page was needed.
    tl.store(OK + batch, accepted)


@triton.jit
def paged_append_kernel(
    KC,
    VC,
    NK,
    NV,
    TABLE,
    LENGTHS,
    OK,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HKV, row % HKV
    # LENGTHS stays immutable during this launch: every head reads the same slot.
    # Advance lengths in a separate kernel, avoiding a cross-head read/write race.
    position = tl.load(LENGTHS + batch)
    accepted = tl.load(OK + batch) != 0
    page = tl.load(TABLE + batch * MAX_PAGES + position // P, accepted, other=0)
    feature = tl.arange(0, BD)
    offset = ((page * HKV + head) * P + position % P) * D + feature
    k = tl.load(NK + row * D + feature, feature < D, other=0)
    v = tl.load(NV + row * D + feature, feature < D, other=0)
    tl.store(KC + offset, k, accepted & (feature < D))
    tl.store(VC + offset, v, accepted & (feature < D))


@triton.jit
def advance_lengths_kernel(LENGTHS, OK, B: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENGTHS + batch, batch < B, other=0)
    accepted = tl.load(OK + batch, batch < B, other=0)
    tl.store(LENGTHS + batch, length + accepted, batch < B)


@triton.jit
def paged_decode_kernel(
    Q,
    KC,
    VC,
    TABLE,
    LENGTHS,
    OUT,
    LSE,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // HQ, row % HQ
    kv_head = head // (HQ // HKV)
    feature, token = tl.arange(0, BD), tl.arange(0, P)
    q = tl.load(Q + row * D + feature, feature < D, other=0)
    length = tl.load(LENGTHS + batch)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0.0, tl.float32)
    numerator = tl.zeros((BD,), tl.float32)
    for logical in range(tl.cdiv(length, P)):
        # A logical consecutive page may be anywhere in physical storage.
        page = tl.load(TABLE + batch * MAX_PAGES + logical)
        t = logical * P + token
        offset = ((page * HKV + kv_head) * P + token[:, None]) * D + feature[None, :]
        valid = (t[:, None] < length) & (feature[None, :] < D)
        k, v = (
            tl.load(KC + offset, valid, other=0),
            tl.load(VC + offset, valid, other=0),
        )
        score = tl.sum(k * q[None, :], axis=1) * (D**-0.5)
        score = tl.where(t < length, score, -float("inf"))
        updated = tl.maximum(maximum, tl.max(score, axis=0))
        correction = tl.exp(maximum - updated)
        weight = tl.exp(score - updated)
        numerator = numerator * correction + tl.sum(weight[:, None] * v, axis=0)
        denominator = denominator * correction + tl.sum(weight, axis=0)
        maximum = updated
    # The loop is skipped for empty sequences; use a safe denominator.
    out = numerator / tl.where(denominator > 0, denominator, 1.0)
    tl.store(OUT + row * D + feature, out, feature < D)
    tl.store(LSE + row, maximum + tl.log(tl.where(denominator > 0, denominator, 1.0)))
