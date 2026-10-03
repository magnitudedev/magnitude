---
"@magnitudedev/cli": patch
---

- Fix models with DFlash2 speculative decoding (Qwen3.8 27B) and Nemotron models failing to load on Vulkan GPUs with a shader compilation error. Every GPU kernel is now compiled for Vulkan, CUDA and Metal before each release, including every kernel each catalog model loads.
