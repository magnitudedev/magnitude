---
"@magnitudedev/cli": patch
---

commit: 44e9fd5d
author: @thrgreenwald

- Fix a request without a `seed` repeating the same sample on every retry, which made some requests (such as Gemma 4 E2B over the Responses API with reasoning off) return an empty answer every time. Each request now samples with a fresh seed unless it names one, and the Responses and Anthropic APIs accept `seed` like Chat Completions.
