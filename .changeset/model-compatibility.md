---
"@magnitudedev/cli": patch
---

author: @thrgreenwald
author: @anerli

Fix incorrect output and load failures for several models.

- Reasoning works with LFM2.5 Instruct and Thinking models, Gemma 4 thinks unless thinking is turned off, and a response that ends inside its reasoning keeps that reasoning instead of failing.
- Default sampling is now temperature 1.0, top_p 0.95 and top_k 20 on the OpenAI and Anthropic APIs alike. The OpenAI APIs used 0.8 and 0.95, and the Anthropic API 1.0 and 1.0, with no top_k on either. The Anthropic API now accepts `top_k`.
- Finetunes run with their base model's vision projector, and Qwen 3.5 models without built-in prediction heads run.
- Fix Vulkan sampling probabilities, Vulkan weight formats failing to compile, and a Metal weight-packing bug.
