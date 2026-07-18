# Native primitive oracle

`llama-parity-oracle` is a thin JSON-lines adapter over the exact pinned
llama.cpp public C API and `common` helpers. It reads one case per stdin line
and writes exactly one evidence record per stdout line. Native logs remain on
stderr.

Request envelope:

```json
{"schemaVersion":1,"caseId":"case-name","operation":"protocol.describe","input":{}}
```

Successful response envelope:

```json
{"schemaVersion":1,"caseId":"case-name","operation":"protocol.describe","status":"ok","evidence":{}}
```

Supported operations are reported by `protocol.describe`. The current operation
set is:

- `configuration.inspect`, `model.metadata`;
- `tokenizer.tokenize`, `tokenizer.token-to-piece`;
- `chat-template.render`, `chat-template.bench`;
- `chat-parser.inspect`, `chat-parser.bench`;
- `grammar.json-schema-to-grammar`;
- `sampler.apply`, `sampler.bench`;
- `reasoning-budget.inspect`;
- `decode.execute-plan`, `decode.abort`, `decode.abort-recovery`;
- `state.execute-script`.

`protocol.describe` itself is also part of the transport contract. The runner
preflights the required operations on both producers and additionally requires
the candidate's live set to match its static capability manifest exactly.

The model and tokenizer operations require `input.modelPath`; artifacts are
resolved and verified by the parity runner, never downloaded by the oracle.
Byte-bearing tokenizer evidence is emitted as integer arrays so invalid UTF-8
cannot be normalized or lost by JSON serialization.

`chat-parser.inspect` accepts optional `chunkPartitions`, where every partition
must concatenate byte-for-byte to `text`. The oracle parses each cumulative
prefix as partial input, parses the final prefix using the requested final mode,
and returns all snapshots plus a `chunkInvariant` result.

The model-backed decode and state operations are deliberately narrow: they do
not tokenize, sample, detokenize, schedule requests, or infer batch contents.
Cancellation is split into an isolated pre-signalled abort and a separate
callback-removal/clear/recovery transition. The reasoning-budget operation
uses the public common sampler with model-tokenized tags and exposes only its
bounded force/last-token observations.

The oracle owns timing boundaries only for the custom sampler, chat-template,
and parser microbenchmarks. Official tools remain the reference for C0/P0,
P1–P4, and the planned P5 batched workload. The oracle does not implement MTMD
or corpus scoring; those descriptors are disabled until the required artifact
and production-API boundaries exist.
