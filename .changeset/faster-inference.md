---
"@magnitudedev/cli": patch
---

commit: b2f79f00
author: @anerli

- Faster local inference across the board: decoding is up to about 3× faster with several requests at once, and prompt processing up to 4.5× faster (Gemma 4 E2B on Apple Silicon), on Apple Silicon and NVIDIA GPUs.
- The one-time optimization after a download now runs in its own process within a fixed time budget, and loading a model never tunes again afterwards. A model also starts optimizing much sooner on first load (Qwen 3.8 27B: 1.9 s instead of 12.6 s).
- Gemma 4 models use their assistant model as a draft head for faster generation.
- Qwen3.5 4B and 9B now use their DFlash drafters for speculative decoding, so they generate considerably faster.
- Faster prefill and decode on Apple Silicon Macs (M1 through M4): weights use a new tiled layout, prompt processing packs tokens, and attention runs on the matrix and scalar units together. The one-time optimization now plans its time across every kernel instead of searching them in order.
- Faster time to the first token for models with a built-in drafting head: each prompt chunk is no longer computed twice.
- Follow-up turns in a conversation reuse the previous turn's prompt even when the template renders past turns differently, so later turns start sooner.
