---
"@magnitudedev/cli": minor
"@magnitudedev/pi-extension": patch
---

Move local-model onboarding into the Magnitude desktop application. The CLI is headless and no longer hosts the interactive onboarding or Magnitude harness. Pi's `/magnitude-setup` opens the desktop, where users choose models and configure their connections, instead of invoking the removed CLI setup flow.

Release the updated Pi extension with the desktop and CLI so new connections install the desktop-compatible handoff.
