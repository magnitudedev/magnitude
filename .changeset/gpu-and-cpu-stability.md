---
"@magnitudedev/cli": patch
---

commit: 7d626619
author: @thrgreenwald

- Fix long agent sessions on Macs running the GPU out of memory, which unloaded the model mid-session. Memory used for earlier turns is now released before the GPU fills.
- Fix NVIDIA GPUs crashing with an illegal memory access when several requests run together after a long prompt.
- Fix large images and long prompts resetting AMD GPUs on Linux and unloading the model: GPU work is now split so no single piece runs past the driver's time limit.
- On computers without a supported GPU, replies now stream as they are written instead of arriving all at once, stopping or abandoning a request no longer holds up the next one, and generation is faster on Intel and other x86 processors.
