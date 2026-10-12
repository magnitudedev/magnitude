---
"@magnitudedev/cli": patch
---

commit: 16021a53
author: @anerli

- Gemma 4 models are now paired with their assistant models in the catalog, so they use them as a draft head for faster generation.
- Embedding tables are streamed instead of held whole in memory, so large models use less memory (Qwen3.8 27B about 600 MB less).
- Faster Gemma decoding after the first-load optimization, and the first load stays within its time budget, so it no longer runs partly unoptimized after an upgrade.
