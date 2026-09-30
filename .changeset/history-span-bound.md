---
"@magnitudedev/cli": patch
---

- Fix long prompts failing partway through with "target graph class ... was not sealed" and the model server going down, as with Qwen3.8-27B on a 64k-token prompt on Apple Silicon. Attention history now stays within the bound its kernels were prepared for on every model, however requests interleave, fork or are reclaimed.
