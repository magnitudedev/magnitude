---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Launching OpenCode 2 with a Magnitude model works again. OpenCode 2 dropped `--model` and shares one background server across launches, so Magnitude now detects the installed version and starts OpenCode 2 on its own server with the model set in its config. OpenCode 1 launches as before.
