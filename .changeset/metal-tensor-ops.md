---
"@magnitudedev/cli": patch
---

- Speed up prompt processing on M5 and later Macs about 2x by running matrix multiplies and attention on the GPU's tensor operations: Qwen3.5-4B at a 64K context now processes prompts at 649 tok/s (previously 308), cutting time to first token from 213 to 101 seconds, with identical output. Other Macs are unchanged.
