---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Overhauled onboarding, model browsing, and the `magnitude models` CLI.

- Added a first-run onboarding flow with live hardware benchmark progress, and replaced the old Discover and Catalog pages with the new **Models** page and **Explore** modal.
- Paged Hugging Face search in both Explore and the CLI now groups results by repository, lets you inspect every quant, and lets you choose or change a model's speculative drafter at download time or from Your Models.
- Updated memory displays across the app and CLI to show hardware-capacity units, placement bars (GPU, RAM, and SSD), and available disk space, and added the ability to hide and re-show models discovered from other apps.
- Replaced `magnitude catalog` commands with `magnitude models` (`models show`, `models pull`, and `models status`), accepting Hugging Face model IDs directly before download.
