---
"@magnitudedev/cli": patch
---

- Keep first-load kernel tuning within its minute for large models too, such as Gemma 4 26B: preparing each kernel's test data now counts against the same time, is done once instead of twice, and is skipped for kernels whose share of the minute cannot cover it, which keep their default configuration.
