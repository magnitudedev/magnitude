---
"@magnitudedev/cli": patch
---

commit: a54897b2
author: @thrgreenwald

- Fix large images failing in vision models. An image that needed more rows than the engine's per-launch limit failed; images now encode up to 4,096 cells, and larger images are resized to fit.
