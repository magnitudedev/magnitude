---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Stability, memory, and backend fixes for the 0.3.0 alpha.

- Fixed Metal working-set headroom enforcement, bank cushion accounting, in-flight memory relief deadlocks, and dense tight-memory step-down (including worker draft-width alignment) so 16 GB and 32 GB Apple Silicon Macs load and run sustained long-context chats reliably.
- Fixed expert scratch lease scoping (`leased lent record` panic), tiered bank pricing before warm-up (`link.rs` warm-cache load panic), and preemption before load preview.
- Fixed Windows WDDM CUDA async submission deadlocks (`cuStreamQuery` flush), skipped CUDA memory probes on unadmitted pre-`sm_80` GPUs, and fixed Vulkan MoE kernel workgroup barrier and subgroup shuffle uniformity on RADV and non-maximal-reconvergence drivers.
- Fixed pure-attention prefix cache retention under tight memory, raw GBNF grammar parsing on thinking models, closed `<think></think>` template extraction in Qwen3-Coder and LFM2/LFM3, and multi-shard GGUF quant labels.
- Fixed web UI unsupported row priority, 0-bit quant labels, and slow-host onboarding fallback; improved CLI GPU diagnostics, `magnitude update check` exit codes, `magnitude hardware` memory rows, and Hugging Face network search errors.
- Added opt-in `--channel alpha` / `-Channel alpha` installer support, fixed Windows `install.ps1` `System32\tar.exe` handling, and degraded non-owner macOS Desktop launches cleanly to read-only.
