# Magnitude inference Python SDK

The ignored package in `generated/` is generated from the same ICN OpenAPI document as the
TypeScript client. It provides synchronous and asynchronous native management operations. The
tracked `streams.py` extension is copied into that generated package and adds typed sync/async SSE
iteration for Chat Completions, Responses, and the multiplexed inference event stream.

Regenerate after changing the ICN contract:

```sh
./python-sdk/generate.sh
```

The generated client defaults are not product lifecycle logic. Python callers address the public
ACN gateway at `http://127.0.0.1:10100/inference`; ACN remains responsible for acquiring and
versioning its private ICN.
