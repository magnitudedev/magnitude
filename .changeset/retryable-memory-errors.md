---
"@magnitudedev/cli": patch
---

commit: cef7f3bb
author: @thrgreenwald

- Fix requests that fail partway through, for example when memory runs short during a long conversation, returning an empty reply that looked like success. They now return a 503 with `Retry-After` and a message agent harnesses recognize, so Pi, OpenCode, Claude Code and others retry them automatically.
- Log memory pressure and request failures from the inference engine, which previously failed silently.
