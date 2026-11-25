---
"@magnitudedev/cli": patch
---

Fix assessment of streamed Gemma 4 26B GGUFs failing with "a plan's tiers hold more than its weights" instead of Fits or DoesNotFit. Placement now counts each row slice of a stored tensor, so fused gate and up experts sum to the tensor's stored bytes.
