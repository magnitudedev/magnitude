---
"@magnitudedev/cli": patch
---

commit: 7aea8365
author: @thrgreenwald

- Fix models failing to load on M1 and M2 Macs with "requests N threads per threadgroup; the pipeline allows M". Metal kernels are now built to accept the thread count they launch with, which fixes Qwen3.8 27B on M1 Max and similar errors in other kernels.
- Fix models with 16 or more query heads per key (Gemma 4 12B, Muse Glimmer 30B, Nemotron 3.5 Lightning, Qwen3.5 122B, Nemotron 3 Super) failing on M1 and M2 Macs. Prefill attention now splits a key's query heads into groups, so it fits every Mac's thread limit, with no change in speed or output elsewhere.
