---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

- Removing a downloaded model no longer unloads and reloads the model in use, which paused its requests and dropped its cached prompts.
- Stopping a model while it is still preparing no longer starts another load.
- Fix the service sometimes stopping responding when the model list was read while a download or removal was being recorded.
- On Linux, a software Vulkan device such as llvmpipe is no longer counted as a GPU.
- After a model fails to load, the message suggests trying again only when trying again can help.
