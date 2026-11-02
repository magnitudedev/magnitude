---
"@magnitudedev/cli": patch
---

author: @anerli

Run more GGUF quantizations, and run several existing ones smaller and faster, on CUDA, Metal, Vulkan and CPU.

- Q1_0, IQ1_S, IQ1_M, IQ2_S and TQ1_0 models now load and run.
- Q2_K, Q3_K, Q4_1, IQ3_S, IQ4_XS, IQ4_NL and TQ2_0 stay at their own size instead of being expanded when loaded. For example, Q3_K goes from 6.56 to 3.44 bits per weight, so larger models fit.
- Decoding is faster for many formats: on CUDA, 13–23% for Q4_1, Q5_0, Q5_1, Q2_K, Q4_K and Q5_K, and 2.5× for IQ3_XXS. On Metal, small batches of Q8_0, Q5_0 and several IQ formats are 29–35% faster. Q3_K prefill on CUDA is 29% faster.
