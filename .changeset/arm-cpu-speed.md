---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Faster CPU inference on ARM. On ARM CPUs with the dot-product instructions, such as AWS Graviton and Ampere, Q8_0 models use new kernels for prompt processing and generation, and Linux ARM64 builds now detect these instructions at run time.

- Prompt processing is 4–13× faster and generation 1.4–5× faster than 0.2.16. For example, Qwen3.5 4B Q4 on Graviton4 goes from 3.8 to 45 tokens/s for prompts and from 3.0 to 14.7 tokens/s for generation.
