---
"@magnitudedev/cli": patch
---

author: @thrgreenwald
author: @anerli

Run custom models: any GGUF of a model family Magnitude supports, not just catalog models, including other quantizations, finetunes, merges and uncensored versions.

- Search Hugging Face and add models from the desktop app, the web app or the CLI (`magnitude catalog search`, `inspect` and `pull`). Every version shows its size and whether it fits your computer, can't run (with the reason) or is gated, before you download it.
- Sign in to Hugging Face once to download gated models; Magnitude says when a model needs sign-in, its terms accepted, or the author's approval.
- Models already on your computer are found and used in place, without copying: the Hugging Face cache, LM Studio, the llama.cpp cache, Ollama, and folders you add. Found models can be hidden without touching their files, and a catalog model already on your computer is used instead of downloaded again.
- Each model gets the best drafter known to work with it, when there is one: from its own repository or folder, the matching catalog model, or its built-in MTP heads. You can choose a different drafter or none.
- Models that can't run are listed with the reason, such as an architecture without kernels yet or an unsupported weight type.
