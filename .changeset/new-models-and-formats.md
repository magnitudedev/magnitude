---
"@magnitudedev/cli": patch
---

author: @anerli

New models, native weight formats, and a refreshed recommended catalog.

- Added support for **DeepSeek V4.1 Flash** (including its tokenizer, tool calling, and attention/engram architecture), **GLM-5.3 Flash** (with vision and multi-token prediction across all backends), and **Nemotron 3.5 Bolt E4B**.
- Added native in-place **FP8** (`F8_E4M3` / `mxfp8g32`) and **GGUF `Q2_0`** weight support across all backends.
- Refreshed the recommended catalog around tested quantization ladders and Epoch Capabilities Index scores (% of frontier), deprecating models superseded by newer ones (Gemma 4 26B-A4B, Gemma 4 31B, Gemma 4 E4B, Liquid LFM2.5 24B-A2B, and Qwen3.5 9B) while continuing to surface and serve already-installed deprecated models.
