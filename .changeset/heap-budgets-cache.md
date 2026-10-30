---
"@magnitudedev/cli": patch
---

Decoding on Linux with NVIDIA GPUs no longer re-enumerates Vulkan devices every step: Qwen3.5 4B Q4 on an A100 went from 35-58 to about 155 tok/s without a drafter.
