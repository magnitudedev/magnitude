---
"@magnitudedev/cli": patch
---

commit: 157ab941
author: @thrgreenwald

- Fix Qwen 3.5, 3.6 and 3.8 models failing with "sampling distribution is nonfinite" on M1 Max and M1 Ultra Macs.
- Fix 8 GB NVIDIA and AMD GPUs unloading the model when several requests run together after a long prompt. Long prompts also go much further on these GPUs (MiniCPM5 2B now handles about 78K tokens instead of failing near 26K).
- Model recommendations and "Fits" labels on NVIDIA and AMD GPUs now match what actually loads, because memory the GPU driver keeps for itself is no longer counted as available.
- Fix requests occasionally failing with "target rows differ" when several run at once.
- Fix every model load crashing on AMD GPUs on Windows.
- Fix long prompts hanging on 8 GB AMD GPUs.
- Computers whose integrated GPU doesn't meet Magnitude's requirements, such as Intel UHD 770, now run models on the CPU instead of failing to load.
