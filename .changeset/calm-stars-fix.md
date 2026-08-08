---
"@magnitudedev/cli": patch
---

Fix first-run install failing with "ACN candidate … no longer available" when daemon startup phases (Resolving / PreparingBackend / Starting) hold a stable progress key longer than 30s
