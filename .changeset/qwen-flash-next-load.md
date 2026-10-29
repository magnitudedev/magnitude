---
"@magnitudedev/cli": patch
---

author: @anerli

Fix Qwen3.8 Flash Next failing to load on Apple silicon with "prepared 105 bytes but the ordered program plan charged 102" (#227). Kernel memory is now derived instead of counted by hand, so the two counts can no longer disagree.
