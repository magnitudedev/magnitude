---
"@magnitudedev/cli": patch
---

Fix a streamed MoE expert-watcher panic (`scratch.rs`) when mid-layer host-record lease extensions evict an expert already routed as a hit in the same prompt group (#292).
