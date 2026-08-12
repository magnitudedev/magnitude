---
"@magnitudedev/cli": patch
---

Add fastCRW as a web search source, so web search can run against a self-hosted
engine with no API key. Set `CRW_API_URL` for a local engine or `CRW_API_KEY` for
the hosted one. Exa keeps precedence when both are configured.

author: @us
