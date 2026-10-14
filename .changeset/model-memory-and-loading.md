---
"@magnitudedev/cli": patch
---

commit: f462efd6b
author: @thrgreenwald

Improve model memory management and loading reliability.

- Keep loaded models' GPU memory resident through idle gaps on Macs, avoiding the delay of wiring it back into memory when the next request arrives.
- Fix large model loads being rejected because of brief memory pressure caused by the load itself. Memory protection now responds to sustained thrashing or swapping.
- Refuse to load a model assessed as not fitting before unloading the current model, and disable its Load action in the desktop.
- Fix NVIDIA GPU capacity being underestimated on Linux when other programs hold GPU memory at startup, so model recommendations and fit assessments use the device's usable capacity.
