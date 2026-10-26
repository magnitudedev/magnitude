---
"@magnitudedev/cli": patch
---

author: @thrgreenwald
author: @anerli

Faster generation.

- On x86 CPUs with AVX2 or VNNI, k-quant, Q8 and 4-bit codebook weights run faster.
- On NVIDIA GPUs, IQ2_XXS weights read faster, and models that stream a token table decode faster: Nemotron 3.5 Bolt on a GB10 went from 36.0 to 41.9 tok/s with drafting off.
- Drafting decides from the time recent steps actually took, so one slow step no longer turns it off, and it resumes once it pays off again.
- AMD GPUs on Linux (Vulkan, RADV) prepare a model's programs in a fraction of the time and memory.
- Chat templates that scan the whole conversation for every message no longer take seconds to render, and short follow-up turns resume from the cached conversation.
- A catalog model's one-time optimization waits until the GPU is free, and loading a model you want takes priority over it.
