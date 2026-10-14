---
"@magnitudedev/cli": patch
---

commit: c57fceadf
author: @thrgreenwald

Improve grammar preparation for structured output and parallel tool calls. Large repetition bounds and shared rules no longer unnecessarily inflate certification and flattening work, and loop handling accounts for overlapping text. Fix JSON-schema additional-property keys to use valid JSON string grammar.
