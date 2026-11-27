---
"@magnitudedev/cli": patch
---

Major performance and memory upgrades for streamed MoE inference across Apple Silicon and NVIDIA GPUs — delivering up to 1.88× faster decode, up to 2.07× faster prefill, and ~2.6 GB lower host memory usage on large MoE models (DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, and GLM-5.3 Flash):

- **Unified VRAM / RAM / SSD expert streaming:** Replaces the old dual-mode streaming path with a single 3-tier pipeline that overlaps resident expert compute with host transfers, uses marginal-value residency, and consolidates staging pools (~2.2 GB host RAM saved).
- **Faster GPU execution:** Adds on-device CUDA host staging, zero-rebind CUDA graph replay, coalesced quantized tile packing across all 29 CUDA formats, Metal command-batch coalescing, and better canonical kernel defaults.
- **Faster storage I/O and scheduling:** Unifies storage and transfer admission with batched Linux `io_uring` direct I/O (with automatic fallback), eliminates super-linear prefix-cache and residency scans, and reclaims ~350 MB of tokenizer host memory.
