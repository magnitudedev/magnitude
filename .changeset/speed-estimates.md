---
"@magnitudedev/cli": patch
---

author: @anerli
author: @thrgreenwald

Speed estimates in Discover are now based on measurements of your own hardware instead of a generic memory-bandwidth figure, so they track real performance more closely. The measurement runs once, in the background, and never while a model is in use.
