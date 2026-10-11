---
"@magnitudedev/cli": patch
---

Stop decoding a zstd Codex request once it passes the 128 MB routing limit, so a few kilobytes of compressed input can no longer expand to gigabytes of memory before being rejected.
