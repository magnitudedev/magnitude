---
"@magnitudedev/cli": patch
---

- Fix the first load of a model taking 20–50 minutes and appearing to hang on CPU-only machines. Kernel tuning now takes at most about a minute on any device, instead of a fixed number of configurations whose time grew with the device's slowness, and its progress reports that minute. On an M4 Max, a first load of Qwen3.5-4B now tunes in about 50 seconds (previously 149 on the GPU and 279 on the CPU) with the same speed afterwards.
- Speed up CPU inference on x86 for models with Q6_K weights by converting their half-precision scales without a slow processor path.
