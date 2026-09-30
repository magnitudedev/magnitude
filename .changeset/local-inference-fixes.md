---
"@magnitudedev/cli": patch
---

- Fix Codex hanging after its first tool call over the Responses WebSocket: follow-up requests now continue from the previous response's output, and request errors end the request instead of leaving it waiting.
- Fix requests that repeat the same image, in one message or across turns, failing with a 400. A repeated image is now encoded once.
- Fix tool call IDs repeating across turns (every turn's first call was `call_0`), which made Claude Code drop tool calls and loop.
- Fix Anthropic token usage counting cached tokens twice in responses and reporting none when streaming, and `count_tokens` requiring `max_tokens`.
- Fix large system prompts being re-read in full when only the last message changes: later requests now resume from the cached prompt.
- Fix tools with free-form object parameters failing on Gemma 4 with "Too many items" (breaking Claude Code and Oh My Pi), Cline failing mid-task with "Output parser would retract a published tool call", and forced tool calls (`tool_choice` "required", "any" or a named tool) repeating until the token limit.
