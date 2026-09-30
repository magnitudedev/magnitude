---
"@magnitudedev/cli": patch
---

- Fix the local model hanging forever when a request arrived while another was generating, as with Qwen3.6 35B-A3B: every later request waited without a response while the model still reported Ready, and the engine held a CPU core at 100%. Admission no longer waits on the running generation, and the engine can no longer wait on work only it could release.
- Fix speculative-decoding models failing mid-request with "only a blocked last page is relocated" during long prompts.
