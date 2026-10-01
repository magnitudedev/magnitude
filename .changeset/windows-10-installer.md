---
"@magnitudedev/cli": patch
---

- Fix the Windows installer failing on Windows 10 with "An interrupted installation could not be recovered (code 4395)". Setup now installs normally and publishes the `magnitude` command to PATH, which also failed on Windows 10 once installation got past that error.
