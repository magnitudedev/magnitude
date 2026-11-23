---
"@magnitudedev/cli": patch
---

author: @anerli

Faster inference and kernel tuning across Metal, CUDA, Vulkan, and CPU.

- Added resident launches on Metal and hardware-derived kernel defaults across Metal, CUDA, and Vulkan, preventing register spills and reducing tuning overhead.
- Unified projection operators across all backends and unified speculative decoding round pricing across model assessment and live draft control.
- Broad kernel speedups across CUDA, Metal, and Vulkan for MLA attention, recurrent chunk scans, vision, readout, routed expert gather/combine, sampling, and draft/corrector passes.
