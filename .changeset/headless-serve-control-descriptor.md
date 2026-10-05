---
"@magnitudedev/cli": patch
---

commit: 2a481f7c
author: @thrgreenwald

- Fix a headless `magnitude serve` that failed to start crashing with EBADF instead of reporting the error that stopped it.
