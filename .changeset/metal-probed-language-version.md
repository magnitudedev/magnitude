---
"@magnitudedev/cli": patch
---

commit: c738ead7
author: @aaronjensen

- Fix models failing to load on some Macs (for example Qwen 3.6 on an M5 Max) with a Metal shader compilation error such as "no template named 'extents' in namespace 'metal'". Metal kernels are now compiled with the same language version the device was probed with, so kernels that use tensor operations build wherever the probe found them available.
