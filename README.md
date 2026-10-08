# Kernel Triton Tutorial and Practice Knowledge Base



## Common kernel patterns you should learn

Once you understand the examples above, most LLM kernels are combinations of a small number of patterns:


| Operation | Main kernel idea |
|---|---|
| ReLU / SiLU / GELU | elementwise |
| bias + activation | fusion |
| Softmax | row reduction |
| RMSNorm | row reduction |
| LayerNorm | multiple row reductions |
| Dropout | elementwise + RNG |
| GEMM | tiled `tl.dot` |
| SwiGLU | GEMM + activation + multiplication |
| Cross entropy | reduction + indexing |
| RoPE | elementwise pairwise rotation |
| Quantization | reductions + packing |
| Dequantization | unpack + scale |
| FlashAttention | tiled GEMM + online Softmax |
| MoE routing | reductions + top-k/scatter |
| KV-cache update | indexed memory operations |
| Decode attention | specialized attention/memory kernel |


## Recommended learning sequence

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
18. Quantized GEMM / FP8
      ↓
19. KV-cache/decode kernels
```