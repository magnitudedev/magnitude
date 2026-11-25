---
"@magnitudedev/cli": patch
---

Fix banked MoE drafter assessment failures, CUDA multi-turn prefill hangs/graph-recording OOM, and idle worker CPU burn.

- Fixed catalog assessment dropping DeepSeek-V4.1-Flash and GLM-5.3-Flash on CUDA, Metal, and Vulkan (`routed_land: launch condition reads policy[10], which was not supplied`, #267) by supplying expert bank policy words in `DraftDemand`.
- Fixed CUDA multi-turn/long-prefill hangs and graph-recording OOM when VRAM is tight by evicting unused cached graphs on `DeviceRefused` / `AllocationCapacity`, falling back to zero-VRAM direct launch, and immediately surfacing uncommitted/partial CUDA submission failures instead of blocking prefill sync.
- Restored the strict `all_served` gate in async-DMA prefill and added non-blocking prefill transport/scratch diagnostics.
- Fixed >100% idle CPU usage on loaded banked models by parking `expert-exchange-pulse` and `expert-exchange-watchdog` threads while no inference steps are in flight.
