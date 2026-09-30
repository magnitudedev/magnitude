---
"@magnitudedev/cli": patch
---

- Fix models that load and run, such as Gemma 4 26B-A4B and Qwen3.6 35B-A3B on Apple Silicon, being reported as unable to run on this computer.
- A model is reported as unsupported only when Magnitude cannot actually run it. A gap in the device's speed measurements now shows "Speed estimate unavailable" instead of hiding the model.
