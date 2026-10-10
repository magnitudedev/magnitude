---
"@magnitudedev/cli": patch
---

commit: c74b8b00
author: @thrgreenwald

- Fix OpenClaw requests hanging: its `automations` tool kept the server busy without answering and could push the loaded model out of memory. These requests now answer in seconds.
- Fix Claude Code and Oh My Pi requests failing with "Too many items" on MiniCPM5 and LFM2.5, and Oh My Pi failing on Qwen3.5 with reasoning off.
- A response cut off by its output limit in the middle of a tool call now ends as an ordinary length stop (`length`, `max_tokens`, or `incomplete`) instead of failing with an error.
- Fix connecting Codex from the Magnitude app on Windows, which always failed.
