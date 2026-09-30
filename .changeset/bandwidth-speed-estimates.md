---
"@magnitudedev/cli": patch
---

- Fix the app getting stuck on "Assessing models" on some hardware: model speed is now estimated from the device's memory bandwidth instead of running kernels on the GPU, which could hang or fail.
