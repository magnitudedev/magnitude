---
"@magnitudedev/cli": patch
---

- Reduce the memory a model needs beyond its weights, so larger models and longer contexts fit: Qwen3.5-4B's working memory fell from 3.7 GB to 120 MB, and the memory reserved to run it from 10.0 GB to 5.9 GB, at the same speed. Image-processing memory is now claimed on the first image and released when idle.
