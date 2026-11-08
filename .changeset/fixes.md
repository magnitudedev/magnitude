---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

- Removing one version of a catalog model no longer deletes the image file shared by another installed version, which left that version unable to read images.
- Concurrent requests to a model that drafts with its own MTP heads no longer fail with "a causal head in a drafting batch has no selection".
- A dropped connection while a download is starting is now retried instead of failing the download, and a failed download logs the reason.
- When a newer release comes out after an update was downloaded, Magnitude offers the newer one and replaces the older download.
- The macOS app icon has its tighter rounded corners back.
