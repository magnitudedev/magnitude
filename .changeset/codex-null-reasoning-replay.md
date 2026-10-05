---
"@magnitudedev/cli": patch
---

commit: 02568d6d
author: @thrgreenwald

- Fix Codex failing to send tool results back to a local model. Reasoning items that a client replays with `content` or `summary` set to null are now accepted as empty.
