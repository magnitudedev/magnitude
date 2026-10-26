---
"@magnitudedev/cli": patch
---

author: @thrgreenwald
author: @anerli

More accurate memory planning and more reliable loading.

- Fit checks count each backend's own memory costs, so a model shown as fitting loads, and a long conversation's cached prompt no longer grows past what was planned.
- Models too large for a small GPU show as not fitting instead of failing to load.
- Magnitude serves on the discrete GPU when your computer also has an integrated one, and shows which device it's using.
- Requested downloads, drafters and projectors included, resume after a restart, and a cancelled download stays cancelled.
- A large image no longer holds a model running on the CPU for minutes.
