---
"@magnitudedev/cli": patch
---

commit: b2f79f00
author: @anerli

- Qwen3.5 4B and 9B now use their DFlash drafters for speculative decoding, so they generate considerably faster.
- Faster prefill and decode on Apple Silicon Macs (M1 through M4): weights use a new tiled layout, prompt processing packs tokens, and attention runs on the matrix and scalar units together. The one-time optimization now plans its time across every kernel instead of searching them in order.
- Faster time to the first token for models with a built-in drafting head: each prompt chunk is no longer computed twice.
- Follow-up turns in a conversation reuse the previous turn's prompt even when the template renders past turns differently, so later turns start sooner.
