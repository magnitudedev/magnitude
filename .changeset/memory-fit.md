---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

More accurate memory estimates on NVIDIA and AMD GPUs, so models fit when they should and don't run out of memory when loaded.

- On CUDA, the estimate counts only memory the allocator can't reuse instead of a fixed 1 GB allowance. Small GPUs fit more: models on an 8 GB card that were limited to 32K now get up to their full 128K context.
- Mixture-of-experts models no longer use gigabytes more GPU memory than their estimate on CUDA and Vulkan.
- On CUDA, memory the driver reserves for running kernels is now counted, so it can't push a loaded model past what was estimated.
