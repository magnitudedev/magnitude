---
"@magnitudedev/cli": patch
---

commit: d5bf92d0
author: @thrgreenwald

- Fix models failing on M1 and M2 Macs during long prompts with "device lost: … Impacting Interactivity", after which the model stayed unloaded. The engine now relaxes the macOS GPU watchdog at start (as llama.cpp does), so prompts of 35k and 69k tokens on Gemma 4 26B complete instead of failing at about 20k.

commit: 048a92de
author: @thrgreenwald

- Fix the Windows engine aborting when a chat template or tool call produced invalid JSON. The template library is now built with C++ exception handling on MSVC, so JSON errors are reported instead of crashing the engine.
