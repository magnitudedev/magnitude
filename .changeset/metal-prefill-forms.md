---
"@magnitudedev/cli": patch
---

- Speed up prompt processing on M5 and later Macs by about 50%: Qwen3.5-4B at a 64K context now processes prompts at about 970 tok/s (previously 652). Attention over the prompt reads keys and values directly through the GPU's tensor operations instead of staging them, and kernel tuning no longer keeps a slower default whose own timing was unstable.
- Speed up prompt processing on every Mac by decoding each block of weights once for up to 512 rows instead of once per 64: matrix multiplies run 7–11% faster on an M4 Pro and 17–25% faster on an M1, with identical output. Qwen3.5-4B at a 64K context processes prompts at 534 tok/s on an M4 Pro (previously 513).
