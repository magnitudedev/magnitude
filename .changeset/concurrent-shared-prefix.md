---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Fix concurrent chats with a shared prompt prefix unloading the model with "target state capacity changed after availability check" (#237, reported by @Wenjix). Requests that continue the same cached prefix now grow together instead of failing every request in flight.
