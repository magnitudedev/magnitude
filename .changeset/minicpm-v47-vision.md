---
"@magnitudedev/cli": patch
---

commit: 707b012cb
author: @anerli

Add MiniCPM-V 4.7 1B and 35B-A3B in Q4, Q5, Q6 and Q8, with image understanding and DSpark speculative decoding. Both models include their 4x vision projectors and Q8 drafters. Image processing reuses identical crops and checks memory capacity before allocating resized images; 16x visual compression is not supported.
