---
"@magnitudedev/cli": patch
---

commit: 539c33338
author: @anerli

Add Qwen3.8 Flash Next in Q2, Q4, Q6 and Q8, with its shared MTP drafting head for faster generation. Its large n-gram tables stream from storage instead of being held whole in memory. Also add a smaller Q2 variant of Qwen3.6 35B A3B, with DFlash speculative decoding.
