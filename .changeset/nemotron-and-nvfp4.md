---
"@magnitudedev/cli": patch
---

author: @anerli

Add Nemotron 3.5 Bolt 74B-A9B, Nemotron 3.5 Super VL 120B-A12B with vision, and Qwen 3.8 Flash Next NVFP4 to the catalog.

- Nemotron 3.5 models draft with LiLiCorr, a new drafting method ranked above DFlash2. On a GB10, Bolt decodes at 55.7 tok/s against llama.cpp's 51.6.
- NVFP4 models that carry NVIDIA's activation scales, such as Qwen 3.8 Flash Next NVFP4 and Super VL's NVFP4 variant, run on FP4 tensor cores on GPUs that have them. On a GB10, Super VL NVFP4 prefills at 826 tok/s against llama.cpp's 552.
- Nemotron 3 Super and Ultra can now run as custom models.
