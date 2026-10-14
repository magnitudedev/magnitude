---
"@magnitudedev/cli": patch
---

commit: 00f20455c
author: @anerli

Improve inference performance through better kernel tuning and token packing.

- The one-time optimization spends its time on the kernels that account for the most measured work, within the existing time budget. This improves generation speed, including Gemma 4 on Apple Silicon.
- Token packing on Apple Silicon now supports 8-bit weight formats, including DFlash draft models, and improves packing accuracy for existing formats. This also speeds up prompt processing.
