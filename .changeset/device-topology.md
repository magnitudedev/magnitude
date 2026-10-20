---
"@magnitudedev/cli": patch
---

commit: 15ff69513
author: @anerli

Fix memory planning on unified-memory and integrated GPUs.

- On NVIDIA GB10 and Apple Silicon, the one-time optimization counts the memory the loaded model already uses instead of overcommitting it (#199, #206).
- Integrated GPUs such as AMD Strix Halo use their shared memory as well as the dedicated carve-out, instead of stopping when the carve-out is full (#207).
- The optimization retries an allocation the GPU refuses instead of failing.
- The app lists each GPU once and shows the backend Magnitude uses for it.
