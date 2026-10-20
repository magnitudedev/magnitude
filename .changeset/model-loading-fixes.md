---
"@magnitudedev/cli": patch
---

commit: 151bf9aef
author: @anerli

Fix models with a Q8_0 output head failing to load, including MiniCPM5 1B Q8 and, with drafting off, Qwen3.6 35B-A3B and Qwen3.5 9B Q6. Faster output for large vocabularies can be selected again during the one-time optimization.
