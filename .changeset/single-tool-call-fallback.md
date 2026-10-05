---
"@magnitudedev/cli": patch
---

commit: 44af293b
author: @thrgreenwald

- Fix Oh My Pi and OpenClaw failing on Qwen models when their tools take free-form JSON among optional properties. Where a model's grammar for parallel tool calls cannot be compiled efficiently but its grammar for a single call can, the request now allows one tool call per turn.
