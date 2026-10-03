---
"@magnitudedev/cli": patch
---

- Speed up generation for mixture-of-experts models with multi-token prediction by drafting three tokens ahead instead of one: Qwen3.6-35B-A3B now generates 106–138 tok/s on a GB10 (previously 97–105) and 109–138 tok/s on an M4 Pro (previously 103–111).
- Speed up generation at a 16K context by about 8% on Macs (67.0 → 72.4 tok/s) and 11% on NVIDIA GPUs (66.3 → 73.4 tok/s), with the same output, by choosing each token while reading less of the output layer and doing more of each step in fewer GPU launches.
- Speed up multi-token prediction on NVIDIA GPUs by loading each expert's weights once per step when several drafted tokens choose it, cutting verification time by up to 11%.
