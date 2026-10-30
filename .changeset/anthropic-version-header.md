---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

The inference engine's Anthropic-compatible messages and token-counting endpoints now accept any `anthropic-version` header value, or no header, instead of rejecting clients that do not send `2023-06-01`.
