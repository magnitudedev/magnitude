---
"@magnitudedev/cli": patch
---

Clamp reported available system memory to physical capacity so model loading does not fail on macOS memory samples that briefly exceed installed RAM.
