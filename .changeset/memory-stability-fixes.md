---
"@magnitudedev/cli": patch
---

author: @anerli

Memory accounting, stability, and platform fixes.

- Improved memory accounting and headroom across platforms: reclaimable macOS file cache and Metal wired ceilings, CUDA discrete VRAM headroom, Vulkan `VK_EXT_memory_budget` caps and command-buffer recycling, and Linux CMA exclusion and transparent huge-page compaction stall fixes.
- Fixed a Windows WDDM circular deadlock during slab growth on CUDA and fixed unwakeable memory waits under fragmented state pools.
- Fixed Windows installer/uninstaller cleanup of pre-existing Magnitude CLI `PATH` entries, added retry/cleanup for transient macOS `hdiutil` busy errors, handled missing `systemctl` on Linux, and added a light rim to the macOS Dock icon.
- Auto-completed onboarding on upgrade when existing models or chats are present, fixed starting new chats on the currently loaded model without swapping it out, and fixed surfacing deprecated/retired catalog models and rejected load errors.
