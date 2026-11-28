---
"@magnitudedev/cli": patch
---

Fix Vulkan and Metal inference stability: expand repeat-block launch counts in `NativeGraphMetadata` (#303, #308, #310, #314), charge first-block Vulkan allocations to the caller entry, and apply the discrete GPU elastic context cache reserve on Vulkan.
