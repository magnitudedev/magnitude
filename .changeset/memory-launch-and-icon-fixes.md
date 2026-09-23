---
"@magnitudedev/cli": patch
---

Clamp reported available system memory to physical capacity so model loading does not fail on macOS memory samples that briefly exceed installed RAM, stop an inherited `ELECTRON_RUN_AS_NODE` (for example from a VS Code terminal) from breaking desktop app launch, and size the macOS app icon to Apple's icon grid so it matches other Dock icons.
