---
"@magnitudedev/cli": patch
---

commit: 5bbdd7380
author: @thrgreenwald

Connect Claude Code without a Claude login. Requests, including those from its built-in subagents, use the loaded local model. If you sign in later, reconnect to use your own login for Claude models.
