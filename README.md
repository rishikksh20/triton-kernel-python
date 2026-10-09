# Kernel Triton Tutorial and Practice Knowledge Base

A hands-on guide to writing GPU kernels in Python with Triton. Start with GPU architecture and vector addition, then build up to reductions, normalization, tiled matrix multiplication, FlashAttention, quantization, decoding, and routed expert kernels.

The repository contains a [GPU fundamentals guide](BASICS.md), **26 self-contained Markdown tutorials**, a [standalone matrix multiplication example](kernels/matrix_multiplication.py), and [Triton-only kernel files](kernels) for the other 25 tutorial topics. The tutorials connect each operation's equations to a PyTorch reference, a Triton kernel, correctness checks, and benchmarks you can run on your GPU.

## Start here

1. Read [GPU architecture and Triton basics](BASICS.md) for the progression from GPU hardware to CUDA execution to Triton programs and tiles. It also covers memory access, masks, numerical correctness, and performance measurement.
2. [Set up a GPU environment](#setup) and run the standalone GEMM example.
3. Work through the [learning sequence](#learning-sequence) in order, starting with [vector addition](tutorials/01_vector_addition.md).

Basic Python and familiarity with arrays are enough to begin reading. The fundamentals guide uses the NVIDIA H100 SXM as its hardware reference; compatible NVIDIA GPUs can run the basic kernels, with tuning choices depending on the device.

## Setup

Running the kernels requires:

- Linux and a Python version supported by your chosen PyTorch and Triton releases.
- A compatible NVIDIA GPU with a working NVIDIA driver.
- CUDA-enabled PyTorch and a compatible Triton installation.

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/rishikksh20/kernel-triton-python.git
cd kernel-triton-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch using the command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for Linux, Pip, Python, and a CUDA version suited to your system. Verify the imports below before changing Triton dependencies. See the [official Triton installation guide](https://triton-lang.org/main/getting-started/installation.html) for installation options, and [the environment setup in BASICS.md](BASICS.md#8-set-up-an-environment) for more detail.

Check the driver with `nvidia-smi`, then verify the Python environment:

```bash
python - <<'PY'
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
python kernels/matrix_multiplication.py
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

Each tutorial states its supported shapes, dtypes, and numerical assumptions. Most use contiguous FP32 tensors; tutorials 18–19 introduce integer formats, packing, and an optional native FP8 GEMM path. Native FP8 in tutorial 19 requires compute capability >= 9.0 and a compatible PyTorch/Triton build; its INT8 path runs independently. Fixed configurations emphasize understanding the algorithms. Tutorials 17 and 23 register custom autograd functions with first-order gradients for attention and dropout respectively; the other raw Triton operations implement forward computation. The FlashAttention examples support the modes described in each guide and use no dropout.

## Correctness and performance

Run each tutorial's PyTorch comparisons and boundary checks before experimenting with tile sizes or launch settings. The benchmark sections produce timings on your own GPU; record the device, package versions, shapes, dtype, and precision settings when comparing results. See [BASICS.md: performance measurement](BASICS.md#15-how-to-measure-performance) for warmup and GPU timing guidance.

The H100 roofline and time-bound graphs in the fundamentals guide are calculated models. To regenerate those SVGs without a GPU, install the plotting dependencies and run:

```bash
python -m pip install matplotlib numpy
python docs/generate_performance_graphs.py
```

The script writes `h100-roofline.svg` and `memory-compute-time-bounds.svg` into [docs/images](docs/images). The other diagrams in that directory illustrate GPU architecture, execution, memory, and tiling.

## Further practice

After completing the tutorials, extend the same patterns to:

| Operation | Main kernel idea |
| --- | --- |
| Group-scaled quantization | Scales per feature group and scaling inside GEMM reductions |
| Paged KV caches | Block-table lookup and cache allocation |
| Split-KV decode | Parallel cache chunks and merging online softmax states |
| Fused training operations | Backward kernels for SwiGLU, RoPE, and cross entropy |
| Expert computation | Per-expert GEMMs, load balancing, and distributed routing |

These are suggested exercises beyond the current 26 tutorials.

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
```

## License

Licensed under [Apache License 2.0](LICENSE).
