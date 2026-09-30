---
"@magnitudedev/cli": patch
---

- Make DFlash, DSpark, and DFlash2 speculative decoding faster than plain decoding on Apple Silicon (Qwen3.6-35B-A3B at 65k tokens: 10.7% faster, previously 7% slower) and faster on NVIDIA (38.8% over plain, previously 28.7%), with better draft acceptance at long context.
- Fix speculative drafts whose layers are all sliding-window (such as Muse-Glimmer's DFlash) failing to load.
- Reduce the time to first token added by speculative decoding from about 2.6% to 0.6% of prompt processing on NVIDIA and from about 1.5% to 0.7% on Apple Silicon.
- Speed up long-context decoding and speculative verification by reading each attention head group's history once, on Apple Silicon, NVIDIA, and Vulkan GPUs.
- Speed up mixture-of-experts decoding on Apple Silicon and prompt processing on NVIDIA.
