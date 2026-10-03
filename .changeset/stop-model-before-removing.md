---
"@magnitudedev/cli": patch
---

commit: cffe46e7
author: @thrgreenwald

- Fix removing a model that is running or loading failing with a misleading error. Removing it now stops the model first, and the confirmation says so.
