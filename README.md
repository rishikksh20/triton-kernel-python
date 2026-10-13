# Kernel Triton Tutorial and Practice Knowledge Base

A hands-on guide to writing GPU kernels in Python with Triton. Start with GPU architecture and vector addition, then build up to reductions, normalization, tiled matrix multiplication, FlashAttention, quantization, decoding, and routed expert kernels.

The repository contains a [GPU fundamentals guide](BASICS.md), **31 self-contained Markdown tutorials**, a [standalone matrix multiplication example](kernels/matrix_multiplication.py), and [Triton-only kernel files](kernels) for the other 30 tutorial topics. The tutorials connect each operation's equations to a PyTorch reference, a Triton kernel, correctness checks, and benchmarks you can run on your GPU.

## Start here

1. Read [GPU architecture and Triton basics](BASICS.md) for the progression from GPU hardware to CUDA execution to Triton programs and tiles. It also covers memory access, masks, numerical correctness, and performance measurement.
2. [Set up a GPU environment](#setup) and run the standalone GEMM example.
3. Work through the [learning sequence](#learning-sequence) in order, starting with [vector addition](tutorials/01_vector_addition.md).

Basic Python and familiarity with arrays are enough to begin reading. The fundamentals guide uses the NVIDIA H100 SXM as its hardware reference; compatible NVIDIA GPUs can run the basic kernels, with tuning choices depending on the device.

## Setup

Running the kernels requires:

- Linux x86_64 and Python 3.12 or 3.13 (the project defaults to 3.12).
- A compatible NVIDIA GPU with a working NVIDIA driver that supports CUDA 12.8.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python and dependencies.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/rishikksh20/kernel-triton-python.git
cd kernel-triton-python
uv sync
```

The small [pyproject.toml](pyproject.toml) manifest installs CUDA 12.8 PyTorch from the official PyTorch wheel index, its compatible Triton version and CUDA runtime libraries, plus NumPy and Matplotlib for the graph generator. `uv` creates `.venv` automatically; activation is optional when using `uv run`. The NVIDIA driver must be installed on the host separately. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for the index configuration and [PyTorch's published CUDA builds](https://pytorch.org/get-started/previous-versions/) for other CUDA versions.

Check the driver with `nvidia-smi`, then verify the Python environment:

```bash
uv run python - <<'PY'
import torch
import triton

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
if not torch.cuda.is_available() or torch.version.hip is not None:
    raise RuntimeError("These examples require a working NVIDIA CUDA GPU.")
print("GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
PY
```

## Run an example

From the repository root, run the standalone matrix multiplication example:

```bash
uv run python kernels/matrix_multiplication.py
```

It computes `(65, 47) @ (47, 70)` with FP16 inputs, compares the result with PyTorch, and prints:

```text
GEMM passed: (65, 47) @ (47, 70) = (65, 70)
```

The `matmul(a, b)` wrapper supports contiguous rank-two FP16 or BF16 tensors with matching dtype on the same CUDA GPU. It accumulates in FP32 and returns the input dtype. Uneven dimensions exercise the masks at tile boundaries.

The tutorials are Markdown files. To run one, copy its numbered Python sections into a fresh notebook and execute them in order, or save its final **Complete code in one block** section as a `.py` file and run it in the same environment. Each tutorial includes its own imports and definitions.

The other files in [kernels](kernels) contain only Triton imports, JIT kernels, and device helpers. Each file documents buffer shapes and dtypes, launch grids, tile sizes, and any required stage ordering or initialization, with comments explaining the computation. Allocate buffers and launch these kernels from your own caller; they contain no PyTorch references, allocation wrappers, benchmarks, or executable examples. The existing matrix multiplication example retains its standalone interface.

## Learning sequence

Each step adds a kernel pattern used by later examples.

| Step | Tutorial | Main kernel idea |
| --- | --- | --- |
| 1 | [Vector addition](tutorials/01_vector_addition.md) | Program IDs, pointer arithmetic, and masked loads/stores |
| 2 | [ReLU / SiLU / GELU](tutorials/02_relu_silu_gelu.md) | Elementwise activation functions |
| 3 | [Fused bias + activation](tutorials/03_fused_bias_activation.md) | Broadcasting and fusion |
| 4 | [Sum reduction](tutorials/04_sum_reduction.md) | Partial sums and multi-stage reductions |
| 5 | [Max reduction](tutorials/05_max_reduction.md) | Partial maxima and padding values |
| 6 | [Softmax](tutorials/06_softmax.md) | Stable row reductions and normalization |
| 7 | [RMSNorm](tutorials/07_rmsnorm.md) | Squared row reduction and feature scaling |
| 8 | [LayerNorm](tutorials/08_layernorm.md) | Mean, variance, and affine transformation |
| 9 | [MatMul](tutorials/09_matmul.md) | Two-dimensional tiles and `tl.dot` |
| 10 | [MatMul + bias + activation](tutorials/10_matmul_bias_activation.md) | Fused output epilogues |
| 11 | [Batched MatMul](tutorials/11_batched_matmul.md) | Independent matrix pairs and a batch grid axis |
| 12 | [Attention without Flash](tutorials/12_attention_without_flash.md) | Separate scores, softmax, and weighted-value kernels |
| 13 | [Online softmax](tutorials/13_online_softmax.md) | Streaming maxima and denominators |
| 14 | [FlashAttention forward](tutorials/14_flashattention_forward.md) | Online normalization and tiled value aggregation |
| 15 | [Causal FlashAttention](tutorials/15_causal_flashattention.md) | Triangular masking and skipping future tiles |
| 16 | [Multi-head FlashAttention](tutorials/16_multi_head_flashattention.md) | Independent state for each batch/head pair |
| 17 | [FlashAttention backward](tutorials/17_flashattention_backward.md) | Probability recomputation, gradients, and PyTorch autograd |
| 18 | [Quantization / dequantization](tutorials/18_quantization_dequantization.md) | Row reductions, signed INT8 encoding, and INT4 packing/unpacking |
| 19 | [Quantized GEMM / FP8](tutorials/19_quantized_gemm_fp8.md) | Low-precision dot products, accumulator precision, and scale epilogues |
| 20 | [RoPE](tutorials/20_rope.md) | Adjacent feature-pair rotations and absolute position indexing |
| 21 | [KV-cache updates](tutorials/21_kv_cache_updates.md) | Indexed appends into preallocated storage |
| 22 | [Decode attention](tutorials/22_decode_attention.md) | Single-query online attention over ragged caches and shared KV heads |
| 23 | [Dropout](tutorials/23_dropout.md) | Seeded random masks and regenerated first-order gradients |
| 24 | [SwiGLU](tutorials/24_swiglu.md) | Two tiled projections with a fused gated epilogue |
| 25 | [Cross entropy](tutorials/25_cross_entropy.md) | Stable log-sum-exp, target indexing, and ignored rows |
| 26 | [MoE routing](tutorials/26_moe_routing.md) | Deterministic top-2 selection, atomic dispatch, and weighted gather |
| 27 | [Group-scaled quantization](tutorials/27_group_scaled_quantization.md) | Scales per feature group and scaling inside GEMM reductions |
| 28 | [Paged KV caches](tutorials/28_paged_kv_caches.md) | Block-table lookup, page allocation, and ordered cache appends |
| 29 | [Split-KV decode](tutorials/29_split_kv_decode.md) | Parallel cache chunks and merging online softmax states |
| 30 | [Fused training operations](tutorials/30_fused_training_operations.md) | Backward kernels for SwiGLU, RoPE, and cross entropy |
| 31 | [Expert computation](tutorials/31_expert_computation.md) | Per-expert GEMMs, load balancing, and distributed routing |

Each tutorial states its supported shapes, dtypes, and numerical assumptions. Most use contiguous FP32 tensors; tutorials 18–19 introduce integer formats, packing, and an optional native FP8 GEMM path, and tutorial 27 adds group-scaled INT8 GEMM. Native FP8 in tutorial 19 requires compute capability >= 9.0 and a compatible PyTorch/Triton build; its INT8 path runs independently. Fixed configurations emphasize understanding the algorithms. Tutorials 17 and 23 register custom autograd functions with first-order gradients for attention and dropout respectively. Tutorial 30 adds raw first-order backward kernels for SwiGLU, RoPE, and cross entropy, with PyTorch autograd comparisons. Tutorial 31 also demonstrates an auxiliary load-balancing loss and its router-logits gradient. Other operations implement forward computation. The FlashAttention examples support the modes described in each guide and use no dropout.

## Correctness and performance

Run each tutorial's PyTorch comparisons and boundary checks before experimenting with tile sizes or launch settings. The benchmark sections produce timings on your own GPU; record the device, package versions, shapes, dtype, and precision settings when comparing results. See [BASICS.md: performance measurement](BASICS.md#15-how-to-measure-performance) for warmup and GPU timing guidance.

The H100 roofline and time-bound graphs in the fundamentals guide are calculated models. To regenerate those SVGs without a GPU, run:

```bash
uv run python docs/generate_performance_graphs.py
```

The script writes `h100-roofline.svg` and `memory-compute-time-bounds.svg` into [docs/images](docs/images). The other diagrams in that directory illustrate GPU architecture, execution, memory, and tiling.

## Advanced extensions

Tutorials 27–31 extend the earlier patterns in this order:

| Operation | Main kernel idea |
| --- | --- |
| [Group-scaled quantization](tutorials/27_group_scaled_quantization.md) | Scales per feature group and scaling inside GEMM reductions |
| [Paged KV caches](tutorials/28_paged_kv_caches.md) | Block-table lookup and cache allocation |
| [Split-KV decode](tutorials/29_split_kv_decode.md) | Parallel cache chunks and merging online softmax states |
| [Fused training operations](tutorials/30_fused_training_operations.md) | Backward kernels for SwiGLU, RoPE, and cross entropy |
| [Expert computation](tutorials/31_expert_computation.md) | Per-expert GEMMs, load balancing, and distributed routing |

These guides include runnable references, boundary checks, and complete code blocks. The paged-cache allocator demonstrates exclusive page ownership with explicit failure status. The expert guide includes a two-rank simulation on one GPU and an optional host collective function for initialized distributed workers; multi-GPU execution requires a configured process group and agreed buffer shapes.

## Recommended Kernel Sequence for Learning :

I would work through Triton in this sequence:

```
1. Vector addition
      ↓
2. ReLU / SiLU / GELU
      ↓
3. Fused bias + activation
      ↓
4. Sum reduction
      ↓
5. Max reduction
      ↓
6. Softmax
      ↓
7. RMSNorm
      ↓
8. LayerNorm
      ↓
9. MatMul
      ↓
10. MatMul + bias + activation
      ↓
11. Batched MatMul
      ↓
12. Attention without Flash
      ↓
13. Online Softmax
      ↓
14. FlashAttention forward
      ↓
15. Causal FlashAttention
      ↓
16. Multi-head FlashAttention
      ↓
17. FlashAttention backward
      ↓
18. Quantization / dequantization
      ↓
19. Quantized GEMM / FP8
      ↓
20. RoPE
      ↓
21. KV-cache updates
      ↓
22. Decode attention
      ↓
23. Dropout
      ↓
24. SwiGLU
      ↓
25. Cross entropy
      ↓
26. MoE routing
      ↓
27. Group-scaled quantization
      ↓
28. Paged KV caches
      ↓
29. Split-KV decode
      ↓
30. Fused training operations
      ↓
31. Expert computation
```

## License

Licensed under [Apache License 2.0](LICENSE).
