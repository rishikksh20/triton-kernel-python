"""Row-scaled symmetric INT8 quantization and packed signed INT4 conversion.

Buffers: contiguous FP32 X/Y (R,C), INT8 Q (R,C), FP32 S (R,).
For INT4, UINT8 P (R,PACKED) stores two signed nibbles per byte,
PACKED=triton.cdiv(C,2). Use R>=1, 1<=C<=4096, R*C < 2**31.
B=triton.next_power_of_2(C). All stages use grid (R,) and num_warps=4.

INT8: quantize_kernel[(R,)](X,Q,S,C,127,B,num_warps=4).
INT4: quantize_kernel[(R,)](X,Q,S,C,7,B,num_warps=4), then
      pack_kernel[(R,)](Q,P,C,PACKED,triton.next_power_of_2(PACKED),num_warps=4).
Decode INT8: dequantize_kernel[(R,)](Q,S,Y,C,C,8,B,num_warps=4).
Decode INT4: dequantize_kernel[(R,)](P,S,Y,C,PACKED,4,B,num_warps=4).
Use BITS=4 or 8 only. The original C and row scales are required metadata.
Inputs must be finite. Quantization rounds ties away from zero and clamps
to [-127,127] or [-7,7]; zero rows use a nonzero scale floor of 1e-8.

Source: tutorials/18_quantization_dequantization.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def quantize_kernel(X, Q, S, C: tl.constexpr, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    # One program derives a scale and quantizes one complete row.
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * C + col, col < C, other=0)
    # Match the row absmax to the signed range; the floor handles zero rows.
    # div_rn requests rounded division for the tutorial rounding convention.
    scale = tl.maximum(tl.div_rn(tl.max(tl.abs(x), axis=0), LIMIT), 1e-8)
    z = tl.div_rn(x, scale)
    # Round halfway values away from zero for both positive and negative x.
    rounded = tl.where(z < 0, -tl.floor(-z + 0.5), tl.floor(z + 0.5))
    # Clamp before converting to INT8; INT4 values are staged in INT8 too.
    q = tl.minimum(tl.maximum(rounded, -LIMIT), LIMIT).to(tl.int8)
    tl.store(Q + row * C + col, q, col < C)
    tl.store(S + row, scale)


@triton.jit
def pack_kernel(Q, P, C: tl.constexpr, PACKED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Each j owns one output byte containing two adjacent signed values.
    j = tl.arange(0, BLOCK)
    # &15 retains the four-bit two's-complement representation.
    # An odd final C uses a zero high nibble.
    low = tl.load(Q + row * C + 2 * j, 2 * j < C, other=0).to(tl.int32) & 15
    high = tl.load(Q + row * C + 2 * j + 1, 2 * j + 1 < C, other=0).to(tl.int32) & 15
    # The even value occupies bits 0..3 and the odd value bits 4..7.
    tl.store(P + row * PACKED + j, low | (high << 4), j < PACKED)


@triton.jit
def dequantize_kernel(
    P,
    S,
    Y,
    C: tl.constexpr,
    STRIDE: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # INT8 is already signed; INT4 needs nibble extraction and sign extension.
    if BITS == 8:
        q = tl.load(P + row * STRIDE + col, col < C, other=0).to(tl.float32)
    else:
        byte = tl.load(P + row * STRIDE + col // 2, col < C, other=0).to(tl.int32)
        # Adjacent columns select the low or high nibble of the same byte.
        nibble = (byte >> (4 * (col % 2))) & 15
        # Four-bit values 8..15 represent -8..-1 in two's complement.
        q = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)
    # Both formats use one scale per original row: reconstructed_x=q*scale.
    scale = tl.load(S + row)
    tl.store(Y + row * C + col, q * scale, col < C)
