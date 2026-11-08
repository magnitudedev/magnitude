---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Run models your computer can't hold at full context. Each model is now served at the largest context that fits in memory, from 32K up to its full length, instead of being refused when its full context doesn't fit.

- Discover recommends the highest-quality version that reaches 64K context, falling back to 32K.
- Chat, the API (`/v1/models`) and connected agents such as Claude Code, Codex and OpenCode use the served context. Claude Code compacts before reaching it instead of assuming 200K. A prompt longer than the served context is refused with `context_length_exceeded`.
- A model's memory breakdown shows the context it's served at when the full context doesn't fit, and `models status` explains why a model doesn't fit.
- A model waits for its memory check before loading, so it never loads at a context the computer can't hold.
