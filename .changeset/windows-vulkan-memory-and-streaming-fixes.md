---
"@magnitudedev/cli": patch
---

Fix Windows memory sizing and iGPU/WDDM support, AMD RDNA3 Vulkan kernel crashes, Engram slab memory blowup, Apple Silicon startup crashes, and streaming decode throughput.

- Fixed Windows CUDA VRAM budgeting via DXGI `IDXGIAdapter3::QueryVideoMemoryInfo`, stopped host-tier resize oscillation from fake trimmed RAM, and rejected mismatched shape-only drafter fallbacks (#269).
- Fixed Windows host-tier startup starvation on tight RAM by reserving drafter head upload before bank growth and classifying host-staging shortfalls as insufficient memory instead of unsupported model (#274).
- Enabled full shared system RAM grants on Windows integrated GPUs (such as Intel Arc 140V and AMD Strix Halo) and displayed shared memory in CLI/UI (#275).
- Fixed Apple Silicon 16 GB startup crash-loops (`assess.rs` fixed-point oscillation) via bracketed bisection when settling bank path records (#272).
- Fixed Windows AMD RDNA3 Vulkan vision/routed NaNs and `gated_delta_project` LLPC crashes by gating optional vision buffers on static presence, adding Vulkan Memory Model device-scope acquire-release semantics, and unifying segmented projection GEMV dispatch (#201, #257).
- Fixed CUDA illegal memory access (700) on vacant decode-row attention tiles, 60s stalls on `DeviceLost`, and `AdoptFence::wait` inside no-blocking-allocation lease paths.
- Fixed Engram recurrent-bank heap blowup (making slab tracking `O(backed)` instead of `O(total_slabs)`), stopped the `split_dma` VRAM-to-NVMe eviction pump (`2×` Qwen3.8 Q2 speedup on RTX 3060, `+28%` DS4 on RTX 5090), and fixed canonical Metal routed projection defaults.
