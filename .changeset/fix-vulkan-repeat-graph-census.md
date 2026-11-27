---
"@magnitudedev/cli": patch
---

Fix a Vulkan long-context prefill panic (`graph_replays.rs`) where `GraphCensus` under-counted repeated launch arguments for multi-round prefill kernels (#303).
