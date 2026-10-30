---
"@magnitudedev/cli": patch
---

author: @anerli

Fix model loading on Metal exceeding its reserved weight-import memory when streaming weight tables (#231). Import batches now share the memory plan used to reserve their budget, accounting for mapped file spans and simultaneous staged uploads.
