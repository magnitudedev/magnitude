---
"@magnitudedev/cli": patch
---

commit: 76d35d73
author: @thrgreenwald

- Fix long requests to a local model failing with a 502 after about five minutes. A non-streaming generation or a long prompt sends nothing until it finishes, and the connection to the engine no longer times out while it waits.
