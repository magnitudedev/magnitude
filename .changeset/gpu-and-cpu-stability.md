---
"@magnitudedev/cli": patch
---

commit: 7d626619
author: @thrgreenwald

- Fix long agent sessions on Macs running the GPU out of memory, which unloaded the model mid-session. Memory used for earlier turns is now released before the GPU fills.
- Fix NVIDIA GPUs crashing with an illegal memory access when several requests run together after a long prompt.
- Fix large images and long prompts resetting AMD GPUs on Linux and unloading the model: GPU work is now split so no single piece runs past the driver's time limit.
- On computers without a supported GPU, replies now stream as they are written instead of arriving all at once, stopping or abandoning a request stops it promptly instead of holding up the next one, and generation is faster on Intel and other x86 processors.
- Fix the one-time optimization never running on Windows, which left every model on slower untuned kernels.
- Fix Qwen3.5 4B and 9B never finishing their one-time optimization on Macs, which re-ran it on every load.
- Fix the optimization failing on GPUs with 8 GB of memory, and on computers without a supported GPU when optimization runs again.
- Restarting Magnitude or the computer no longer repeats the optimization on the next load.
- The first use of a model on AMD GPUs on Linux is much faster: the optimization now prepares every kernel instead of leaving them to the first load.
- Fix image requests failing with "insufficient memory" after a long session on AMD GPUs on Linux.
- Fix Magnitude closing at launch when it is opened while a previous instance is still quitting, or while it takes over from `magnitude serve`.
- `magnitude models load` no longer reports a transport error when a model takes longer than five minutes to load.
- When the inference engine stops unexpectedly, its exit code and log now appear in Magnitude's logs.
