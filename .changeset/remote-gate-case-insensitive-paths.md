---
"@magnitudedev/cli": patch
---

Gate remote callers in the /rpc and inference route handlers instead of the middleware, so case, slash, and percent-encoded path variants can no longer skip the API key check.
