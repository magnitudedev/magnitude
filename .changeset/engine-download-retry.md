---
"@magnitudedev/cli": patch
---

author: @thrgreenwald

Downloading the inference engine survives outages and slow connections. Each part retries until 10 minutes pass without data (it gave up after two quick retries before), partial downloads resume after a restart instead of starting over, and the service keeps retrying a failed download while `magnitude status` shows the error and when it tries again. Startup is no longer cancelled after five minutes while a download is still progressing.
