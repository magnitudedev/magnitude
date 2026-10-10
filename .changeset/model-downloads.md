---
"@magnitudedev/cli": patch
---

commit: 9452ff4e
author: @thrgreenwald

- A download that stops receiving data now reports that it can't reach the model source and offers Retry, which resumes from where it stopped, instead of freezing.
- Clicking Download right after cancelling a download now starts it again instead of doing nothing.
- An expired or revoked `HF_TOKEN` no longer blocks downloads of public models.
- Removing a model on Windows now deletes its files and frees the disk space.
