# GLM-5.3-Flash on HPU (experimental)

This branch adds an eager text-generation adapter for the official GLM-5.3-Flash architecture. It reuses upstream model classes and weight loading. It retains the learned sparse indexer, four-token pooling, partial-pool tails, KDA recurrence, hyper-connection normalization, and clamped expert activation.

The adapter currently requires eager execution, prefix caching disabled, and no speculative decoding or expert parallelism. Set `PT_HPU_LAZY_MODE=0`, `VLLM_FLATTEN_INPUT=true`, `VLLM_CONTIGUOUS_PA=0`, `VLLM_KDA_SCAN=seq`, and `VLLM_GDN_COMPUTE_FP32=1`. Set an explicit `--num-gpu-blocks-override` sized against actual allocations: the hybrid cache planner's logical block budget has not yet been qualified for this layout.

Raw keys and gates occupy a padded BF16 cache with the same page geometry as the latent cache. Pooling and exact E4M3FN-value quantization run on HPU during selection; model values stay on HPU, while scheduling metadata is read on the host. This uses more cache memory and host synchronization than a compressed-cache implementation. It is a correctness baseline, not a tuned serving profile.

The Hadamard/quantization helper intentionally breaks compilation: the tested HPU compiler removes required BF16 rounding in a FP32-to-BF16-to-FP32 round trip. Do not remove this boundary without exact quantization tests.

Validation includes pooled-selection causality and tails, a CPU sequential recurrence oracle for KDA, and complete small-architecture generation with seeded test weights. The small architecture includes KDA, sparse MLA, MoE, and mHC. The official FP8 checkpoint also loaded with TP4 on Gaudi2 and passed arithmetic streaming, a tool-call round trip, and a retrieval prompt beyond the sparse top-k length. These smoke tests do not establish broad quality parity, vision support, or production performance. The manual serving card is `tests/models/language/generation/model_cards/GLM-5.3-Flash.yaml`; no lm-eval accuracy baseline is claimed.

Single-token KDA uses the direct recurrence instead of constructing and solving a one-token chunk. Prefill retains the reviewed chunk operator. Both paths are checked against the sequential CPU oracle.

The fused HPU expert operator omits the model's SwiGLU clamp. Clamped BF16 experts reuse the unfused activation path; clamped FP8 experts use the existing gathered-expert implementation with bounded token slices. The FP8 path requires dynamic per-channel weights. The plugin's default block-to-channel weight conversion changes checkpoint quantization and still needs model-quality validation. A regression deliberately crosses clamp boundaries and checks both clamped and unclamped execution.

The KDA chunk operator is reused from `abotsis/vllm-gaudi` commit `a35e515b13373860da43c1f58bf1c07e776a9117`, retaining its Apache-2.0 header. The NoPE correction follows upstream PR #1795. No indexer checkpoint weights are discarded.
