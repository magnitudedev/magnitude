---
"@magnitudedev/cli": patch
---

commit: 835a4761
author: @thrgreenwald

- Fix an OpenCode session being rejected with "assistant content is required unless tool_calls are present" after a step that failed before producing any output. An empty assistant turn in the history is now skipped, in both the Chat Completions and Anthropic APIs.
