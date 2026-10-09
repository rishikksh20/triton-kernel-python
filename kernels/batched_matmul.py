"""Batched GEMM: C[b] = A[b] @ B[b], with no batch broadcasting.

Buffers: contiguous FP32 A (BATCHES,M,K), B (BATCHES,K,N), C (BATCHES,M,N).
Launch: batched_matmul_kernel[(triton.cdiv(M,32), triton.cdiv(N,32), BATCHES)](
    A, B, C, M, N, K, BLOCK_M=32, BLOCK_N=32, BLOCK_K=32, num_warps=4)
BATCHES <= 65535 for grid axis 2. Skip empty batch/row/column dimensions.
K=0 leaves each accumulator at zero. Each batch has independent storage.

Source: tutorials/11_batched_matmul.md

Allocate buffers on the same CUDA device and validate the contracts above
before launching. These raw kernels do not allocate or validate host tensors.
Pointer strides count elements, not bytes; constexpr arguments specialize
compiled code. Tile sizes using tl.arange must be powers of two.
"""

import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # The two-dimensional grid assigns a unique output tile to each program.
    # Grid axis 2 selects one independent A/B pair.
    # batch*M*K and batch*K*N skip entire contiguous matrices.
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    columns = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    local_k = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(tl.cdiv(K, BLOCK_K)):
        k_indices = k_tile * BLOCK_K + local_k

        # Contiguous A uses row stride K; contiguous B uses row stride N.
        a_ptrs = A + batch * M * K + rows[:, None] * K + k_indices[None, :]
        b_ptrs = B + batch * K * N + k_indices[:, None] * N + columns[None, :]
        a_valid = (rows[:, None] < M) & (k_indices[None, :] < K)
        b_valid = (k_indices[:, None] < K) & (columns[None, :] < N)
        a_tile = tl.load(a_ptrs, mask=a_valid, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_valid, other=0.0)

        # FP32 products and accumulator; choose precision explicitly.
        accumulator = tl.dot(a_tile, b_tile, accumulator, input_precision="ieee")

    c_ptrs = C + batch * M * N + rows[:, None] * N + columns[None, :]
    c_valid = (rows[:, None] < M) & (columns[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_valid)
