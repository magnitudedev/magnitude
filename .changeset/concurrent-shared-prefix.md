---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Fix concurrent chats with a shared prompt prefix unloading the model with "target state capacity changed after availability check" (#237, reported by @Wenjix). Requests that continue the same cached prefix now grow together instead of failing every request in flight.

Concurrent requests that share a long prompt, such as an agent's parallel subagents, are also much faster:

- A request now waits for another request that is already computing their shared prefix and reuses it, instead of computing it again. With a draft model, 8 concurrent requests sharing a 15K-token prompt went from 226 s each to 3.5 s on an NVIDIA A10G.
- A shared prefix stays cached after a batch finishes, so repeating it is fast: 4 concurrent requests on a 13.6K-token prompt went from 12 s to 0.7 s, and 8 from 24 s to 1.2 s.
- Long conversations running side by side recompute their history from scratch less often.
