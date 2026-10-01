---
"@magnitudedev/cli": patch
---

commit: e38af0e9
author: @thrgreenwald

- Fix models failing to load on M1 and M2 Macs when a kernel's default configuration needs more threads than the chip allows for it. Tuning now starts from the nearest configuration that runs.
