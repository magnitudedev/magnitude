---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

See what a model's memory is made of: weights, context and overhead.

- In the app, the info icon next to a model's memory on its card, and next to Fit on the Discover radar, shows the breakdown.
- `magnitude catalog show`, `magnitude models status` and their `--json` output include the breakdown. `magnitude hardware` shows a loaded model's weights, context and overhead separately, counted the same way as in Settings.
