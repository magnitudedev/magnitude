---
"@magnitudedev/cli": patch
---

author: @anerli

Fix slow decoding on Linux with NVIDIA GPUs (#235, reported by @puhuk). Every memory check during decoding read the memory budget of every GPU through every backend, re-enumerating Vulkan devices each time, even when the model ran on CUDA. Memory checks now read only the device in use, and Vulkan budgets are read from the device handle found at discovery. On two A100s with the reported driver (580.178.04), a Qwen3.5 4B decode step went from 13.0 to 6.7 ms, and prefill is 32% faster.
