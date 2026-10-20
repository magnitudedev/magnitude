---
"@magnitudedev/cli": patch
---

commit: 7ec93279c
author: @thrgreenwald

Fix model loads failing on Linux machines and virtual machines (#196).

- Memory held by the file cache counts as available, so loads no longer fail for lack of memory that the system could free.
- A real memory shortage at startup is reported as one that can be retried, instead of failing the load partway through.
- A CPU without the instructions the engine needs is reported plainly before loading starts.
