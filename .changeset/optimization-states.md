---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Fix a model staying slower than it should when it was used while its one-time optimization was running after download. Loading it, chatting with it in the app, or sending it a request from the API or an agent stopped the optimization part way, and the model kept the untuned settings. The optimization now finishes, either during that load or later when the device is free.

- Discover and My Models show "Optimizing" while the optimization runs, and "Optimization pending" while it waits to run when no model is loaded.
- After this update, each downloaded model optimizes once more, so its next optimization or first load takes about a minute longer.
