---
applies_to:
  - inference/catalog/**
  - inference/crates/icn-models/**
  - inference/crates/icn-speculative/**
---

# Catalog speculative methods

`Current method` is what Magnitude activates. `Current method approach` says whether its draft
capability is embedded in the target or supplied by a separate file. `Draft model` links to the
configured artifact; `Best known method` intentionally contains no links.

| Model | Best known method | Current method | Current method approach | Draft model |
| --- | --- | --- | --- | --- |
| Qwen3.5 4B | MTP | MTP | Embedded | [Target model](https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF) |
| Qwen3.5 9B | MTP | MTP | Embedded | [Target model](https://huggingface.co/unsloth/Qwen3.5-9B-MTP-GGUF) |
| Qwen3.6 27B | DFlash | DFlash | Separate file, draft repo | [Qwen3.6-27B-DFlash-Q8_0.gguf](https://huggingface.co/magnitudedev/Qwen3.6-27B-DFlash-GGUF/blob/main/Qwen3.6-27B-DFlash-Q8_0.gguf) |
| Qwen3.8 27B | MTP | MTP | Embedded | [Target model](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| Qwen3.6 35B-A3B | DFlash | DFlash | Separate file, draft repo | [Qwen3.6-35B-A3B-DFlash-Q8_0.gguf](https://huggingface.co/magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF/blob/main/Qwen3.6-35B-A3B-DFlash-Q8_0.gguf) |
| Muse Glimmer 30B | DFlash | DFlash | Separate file, target repo | [dflash-kquant.gguf](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/blob/main/dflash-kquant.gguf) |
| Gemma 4 E2B | MTP | MTP | Separate file, target repo | [mtp-gemma-4-E2B-it.gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/blob/main/mtp-gemma-4-E2B-it.gguf) |
| Gemma 4 E4B | MTP | MTP | Separate file, target repo | [mtp-gemma-4-E4B-it.gguf](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/blob/main/mtp-gemma-4-E4B-it.gguf) |
| Liquid LFM2.5 2.6B | None known | None | — | — |
| Liquid LFM2.5 8B-A1B | None known | None | — | — |
| Bonsai 8B 1-bit | None known | None | — | — |
| Gemma 4 12B | MTP | MTP | Separate file, target repo | [mtp-gemma-4-12B-it.gguf](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/blob/main/mtp-gemma-4-12B-it.gguf) |
| Gemma 4 26B-A4B | MTP | MTP | Separate file, target repo | [mtp-gemma-4-26B-A4B-it.gguf](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF/blob/main/mtp-gemma-4-26B-A4B-it.gguf) |
| Gemma 4 31B | MTP | MTP | Separate file, target repo | [mtp-gemma-4-31B-it.gguf](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/blob/main/mtp-gemma-4-31B-it.gguf) |
| Laguna S 2.1 118B-A8B | None known | None | — | — |
| Qwen3.5 122B-A10B | MTP | MTP | Embedded | [Target model](https://huggingface.co/unsloth/Qwen3.5-122B-A10B-MTP-GGUF) |
| Nemotron 3 Super 120B-A12B | MTP | None | — | — |
| Nemotron 3.5 Lightning 30B-A3B | DFlash | DFlash | Separate file, draft repo | [NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash.gguf](https://huggingface.co/magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash-GGUF/blob/main/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash.gguf) |
| DeepSeek V4 Flash 284B-A13B | DSpark | DSpark | Separate file, target repo | [dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf) |
| Nemotron 3 Ultra 550B-A55B | MTP | None | — | — |
| GLM 5.2 753B-A40B | MTP | MTP | Embedded | [Target model](https://huggingface.co/unsloth/GLM-5.2-GGUF) |

Update this table whenever a catalog model or its configured speculative method changes.
