---
"@magnitudedev/cli": patch
---

Use the sliding-window history limit when backing and reclaiming request memory, so long prompts are not rejected for requesting window storage they do not need.
