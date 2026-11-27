---
"@magnitudedev/cli": patch
---

Fix excessive GPU memory allocation on large resident MoE models and non-finite outputs on long prompts (#294, #238):

- **Bounded whole-model prefill workspace arenas:** Caps merged prefill pass sizing for resident MoE models to tile saturation instead of scaling linearly with full context length, eliminating 16–57 GB of unused workspace/graph allocations on 64K–256K context models (such as DeepSeek-V4.1-Flash on 256 GB Apple Silicon and Qwen3.8-Flash-Next NVFP4 on 128 GB Apple Silicon).
- **Consistent merged prompt-group cancellation:** Shares pulse-expiry and taint state across all steps of a scratch-served merged prompt group so mid-group interruptions cannot produce non-finite outputs or poisoned prefix cache checkpoints.
