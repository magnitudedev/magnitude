---
"@magnitudedev/cli": patch
---

commit: 541ccf13
author: @thrgreenwald

- Magnitude writes logs again: `service.log`, `inference.log` and `desktop.log` in `~/.magnitude/logs` record the service, the inference engine and the app, including crashes and restarts. Each file is size-capped. Attach them when reporting a problem.
