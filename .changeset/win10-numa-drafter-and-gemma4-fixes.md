---
"@magnitudedev/cli": patch
---

Fix Windows 10 startup NUMA discovery, catalog drafter attribution, and streamed Gemma 4 CUDA startup panics.

- Fixed startup panic on Windows 10 (builds prior to 20348) by requesting `RelationNumaNode` instead of `RelationNumaNodeEx` during host NUMA discovery (#277, #279).
- Fixed catalog drafter downloads (such as DSpark) being misattributed as their target model by including architecture in package identity and repairing stores that recorded one on load (#280).
- Fixed CUDA startup assessment panic on streamed Gemma 4 26B-A4B partial expert banks by deriving each block's served decode split once and carrying it across certification, graphs, and runtime (#276).
