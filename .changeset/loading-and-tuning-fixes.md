---
"@magnitudedev/cli": patch
---

commit: 7f65eb32
author: @anerli

- macOS loads a model that fits beside wired and compressed memory instead of refusing it, including right after a download.
- Qwen 3.8 27B loads on NVIDIA GPUs, and Gemma 4 E2B and E4B no longer hit an illegal memory access on NVIDIA GPUs.
- The one-time optimization is more reliable: it no longer settles on slow defaults for large contexts (Gemma 4 12B at a 65,536-token context decoded at 4.8 tok/s), tunes sliding-window layers at the history they keep, and finishes in bounded time on first load.
- Muse Glimmer can turn reasoning off.
