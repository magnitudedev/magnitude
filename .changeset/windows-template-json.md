---
"@magnitudedev/cli": patch
---

commit: 048a92de
author: @thrgreenwald

- Fix the Windows engine aborting when a chat template or tool call produced invalid JSON. The template library is now built with C++ exception handling on MSVC, so JSON errors are reported instead of crashing the engine.
