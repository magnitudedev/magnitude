---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Standalone CLI-only installation and smoother headless serving.

- Added a CLI-only install path for running Magnitude without the desktop app, with `magnitude app open` opening the browser UI (or printing its URL when running headless or over SSH) and `magnitude update install` handling self-updates.
- Improved handoff between `magnitude serve` and the desktop app: installing or starting the desktop app cleanly yields an active `magnitude serve`, waiting `serve` instances stay alive through updates, and `magnitude serve` restores previously loaded models when taking back over.
- Removed the `magnitude://` protocol handler and its OS registrations across all platforms.
