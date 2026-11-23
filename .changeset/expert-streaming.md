---
"@magnitudedev/cli": minor
---

author: @anerli

Stream mixture-of-experts (MoE) models that are larger than your GPU memory or system RAM. Magnitude now keeps hot routed experts resident in the fastest available tier and streams the rest across GPU memory, host RAM, and disk as tokens are generated.

- Models that previously wouldn't fit in VRAM or unified memory can now load and run by caching active experts in GPU/RAM and reading colder experts from host RAM or SSD on demand.
- Streaming adapts automatically while serving, balancing expert cache residency, prompt processing, and KV cache room based on measured hardware and decode timings.
- The app's Status card, Models page, and CLI show how a model is placed across GPU memory, host RAM, and SSD, including live expert residency while running.
