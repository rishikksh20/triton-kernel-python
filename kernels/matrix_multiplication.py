"""Small, forward-only GEMM: C = A @ B (alpha=1, beta=0).

A is (M, K), B is (K, N), and C is (M, N).
Inputs must be contiguous FP16/BF16 tensors on the same CUDA GPU.
Run the example: python3 kernels/matrix_multiplication.py
Reference: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
"""

import torch
import triton
import triton.language as tl


# Python launches this function; @triton.jit compiles its body for the GPU.
# A, B, C are device pointers. BLOCK is known at compile time because Triton
# needs fixed tile shapes to distribute the work across GPU execution lanes.
@triton.jit
def matmul_kernel(A, B, C, M, N, K, BLOCK: tl.constexpr):
    # Each program computes one BLOCK x BLOCK output tile. A program ID
    # identifies a tile, not a CUDA thread. Grid axis 0 selects rows;
    # axis 1 selects columns. Different programs write disjoint output tiles.
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    offsets = tl.arange(0, BLOCK)

    # C[i, j] = sum_k A[i, k] * B[k, j]. Keep partial sums in FP32 to
    # reduce rounding error; the final result uses the input dtype.
    acc = tl.zeros((BLOCK, BLOCK), dtype=tl.float32)
    for start in range(0, tl.cdiv(K, BLOCK)):
        ks = start * BLOCK + offsets

        # Contiguous row-major storage: address(i, j) = base + i*width + j.
        # Strides here count elements, not bytes. A has width K; B has width N.
        # [:, None] makes a column vector; [None, :] makes a row vector.
        # Broadcasting their sum creates a BLOCK x BLOCK grid of addresses.
        a_ptrs = A + rows[:, None] * K + ks[None, :]
        b_ptrs = B + ks[:, None] * N + cols[None, :]

        # M, N, or K may not divide evenly into tiles. Masks prevent invalid
        # reads. Replacing missing inputs with zero preserves the dot product,
        # including the partial tile at the end of the reduction dimension K.
        a = tl.load(a_ptrs, (rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, (ks[:, None] < K) & (cols[None, :] < N), other=0.0)

        # Matrix multiply, not elementwise multiply: a is (rows, ks),
        # b is (ks, cols). tl.dot adds this K slice's contribution to acc.
        acc = tl.dot(a, b, acc)

    # Store only actual output positions. C has width N; tl.store converts
    # FP32 sums to C's dtype. No atomics are needed: each tile has one owner.
    c_ptrs = C + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc, (rows[:, None] < M) & (cols[None, :] < N))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Allocate C and launch the GPU kernel; this wrapper runs in Python."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("Expected A(M, K) and B(K, N).")
    if not (a.is_cuda and b.is_cuda and a.device == b.device):
        raise ValueError("Inputs must be CUDA tensors on the same device.")
    if a.dtype != b.dtype or a.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Inputs must have the same FP16 or BF16 dtype.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("Inputs must be contiguous for row-major indexing.")

    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    if c.numel() == 0:
        return c

    # Ceiling division covers partial edge tiles: M=65 needs three row tiles.
    # BLOCK=32 fixes all three tile dimensions; num_warps controls execution
    # resources, not tile dimensions. The first launch compiles the kernel.
    block = 32
    grid = (triton.cdiv(m, block), triton.cdiv(n, block))
    with torch.cuda.device(a.device):
        matmul_kernel[grid](a, b, c, m, n, k, BLOCK=block, num_warps=4)
    return c


if __name__ == "__main__":
    # Uneven dimensions exercise masks along M, N, and K. FP32 accumulation
    # still allows small rounding differences, so compare with a tolerance.
    torch.manual_seed(0)
    a = torch.randn((65, 47), device="cuda", dtype=torch.float16)
    b = torch.randn((47, 70), device="cuda", dtype=torch.float16)
    result = matmul(a, b)
    torch.testing.assert_close(result, a @ b, rtol=1e-2, atol=1e-2)
    print(f"GEMM passed: {tuple(a.shape)} @ {tuple(b.shape)} = {tuple(result.shape)}")
