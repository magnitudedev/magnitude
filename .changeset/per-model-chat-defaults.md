---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Save chat settings as defaults for each model (#234). Chat settings now show whether a chat uses the standard settings, your defaults for its model, or has changed from them, with Save as default, Revert and Reset to standard. New chats and model switches start from that model's defaults, including the thinking level, system prompt, sampling settings, seed and stop sequences, and the defaults are kept across restarts and updates.
