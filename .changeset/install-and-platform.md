---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Fix installation and platform issues.

- On Windows 10, upgrades no longer fail to replace the installed app.
- The desktop app no longer takes over a Magnitude server running as a Windows service, and says why.
- Hermes connections on Windows are written where Hermes reads them.
- Fix installer readiness checks, macOS onboarding, and verification error messages.
- CLI commands piped into a reader that stops early end quietly instead of crashing.
