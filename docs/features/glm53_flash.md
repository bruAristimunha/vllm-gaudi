# GLM-5.3-Flash on HPU (experimental)

This branch adds an eager text-generation adapter for the official GLM-5.3-Flash architecture. It reuses upstream model classes and weight loading. It retains the learned sparse indexer, four-token pooling, partial-pool tails, KDA recurrence, and hyper-connection normalization.

The adapter currently requires eager execution, prefix caching disabled, and no speculative decoding or expert parallelism. Set `PT_HPU_LAZY_MODE=0`, `VLLM_FLATTEN_INPUT=true`, `VLLM_CONTIGUOUS_PA=0`, `VLLM_KDA_SCAN=seq`, and `VLLM_GDN_COMPUTE_FP32=1`. Set an explicit `--num-gpu-blocks-override` sized against actual allocations: the hybrid cache planner's logical block budget has not yet been qualified for this layout.

Raw keys and gates occupy a padded BF16 cache with the same page geometry as the latent cache. Pooling and exact E4M3FN-value quantization run on HPU during selection; model values stay on HPU, while scheduling metadata is read on the host. This uses more cache memory and host synchronization than a compressed-cache implementation. It is a correctness baseline, not a tuned serving profile.

The Hadamard/quantization helper intentionally breaks compilation: the tested HPU compiler removes required BF16 rounding in a FP32-to-BF16-to-FP32 round trip. Do not remove this boundary without exact quantization tests.

Validation so far includes pooled-selection causality and tails, a CPU sequential recurrence oracle for KDA, and complete small-architecture generation with seeded test weights. The small architecture includes KDA, sparse MLA, MoE, and mHC. These tests do not qualify official checkpoint quality, FP8 loading, tensor parallelism, vision, or performance.

The KDA chunk operator is reused from `abotsis/vllm-gaudi` commit `a35e515b13373860da43c1f58bf1c07e776a9117`, retaining its Apache-2.0 header. The NoPE correction follows upstream PR #1795. No indexer checkpoint weights are discarded.
