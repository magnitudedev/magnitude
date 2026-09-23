---
"@magnitudedev/cli": patch
---

Require the network access API key for remote inference requests regardless of URL path case, so `/INFERENCE/...` no longer skips the check.
