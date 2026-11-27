---
"@magnitudedev/cli": patch
---

Fix Vulkan device initialization failing on GPUs where memory type 0 cannot back storage buffers (such as Intel Arc Pro B70 on Mesa ANV) by resolving allocations against buffer `memoryTypeBits` (#297).
