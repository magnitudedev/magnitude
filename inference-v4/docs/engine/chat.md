# Chat preparation and serving

**One prepared chat request determines prompt, generation prefix, parser, and grammar.**
Transport adapts the shared engine; it does not introduce another inference path.

## Ownership

| Owner | Responsibility |
| --- | --- |
| Host preparation | Request validation, template selection, media preparation, tokenization, and symbolic grammar plan |
| Native template/parser library | Rendering, template analysis, grammar construction, and semantic stream parsing |
| Execution owner | Exact-vocabulary grammar binding, request-local matcher, and generation |
| Host stream | Incremental decoding, string stops, reasoning/tool events, and response framing |

Reuse the pinned standalone native template/parser extraction with its maintained
patches and ownership-safe bindings. This does not require llama.cpp inference or
GGML. Token constraints use llguidance with equivalent grammar conversion semantics.

## Preparation

1. Validate supported request options and tool/schema capabilities.
2. Resolve artifact template, effective tools, and reasoning controls.
3. Prepare bounded media with artifact-defined processing.
4. Produce one consistent native plan, expanded prompt, tokens, and grammar.
5. Bind the grammar against the exact vocabulary before numerical admission.

| Policy | Required meaning |
| --- | --- |
| Template selection | Operator override, explicit variant, effective tool-use variant, and declared default have deterministic precedence |
| Thinking | Omitted controls preserve authored defaults; unsupported effort and conflicting controls fail |
| Tools | Auto, none, required, named selection, and parallel-call policy agree in prompt and grammar |
| Caller history | Normalization preserves caller-owned input rather than mutating it |
| Schemas | Enforce the supported language; reject constraints the selected handler cannot enforce |
| Tokenization | Preserve byte-BPE vocabulary, normalization, special-token policy, EOS identities, padding, and incremental UTF-8 |

Template/parser family support does not imply numerical model-family support.
Exact prompt counting uses the same preparation path without executing model weights.

The wire boundary rejects unknown fields and unsupported generation policies before
native preparation. It accepts one completion with temperature zero or one and
unmodified logits; token-limit aliases must agree. Stop strings are bounded by
Unicode character count. Named/required tools and JSON response formats are checked
against effective offered tools. Model identity and rendered context limits come
from host configuration, not request-controlled capacity fields. Media content must
reach a supported media preparation path; it cannot silently become text-only input.

## Streaming

```text
published tokens → incremental text → bounded stop filter
    → native semantic parser → content / reasoning / tool events → HTTP/SSE
```

- Parsing is independent of chunk boundaries.
- Tool IDs remain stable; starts and argument deltas are append-only.
- Natural completion, user stop, length, cancellation, and failure remain distinct.
  Truncation cannot fabricate a completed structured tool call.
- Stop matching retains bounded lookbehind across chunks.
- Disconnect releases the receiver and cancels generation without abandoning
  already-submitted work or late preparation results.
- Native parsers remain host-local. A unique request receiver awaits publication
  notifications; cancelling an active receive releases its request.
- Accepted output drains before terminal failure is reported, including after
  execution-owner teardown. String stops end parsing within the current publication
  and release remaining output. User stops take precedence over identical template
  stops; template stops preserve natural completion.
- SSE frames encode semantic events as JSON with stable tool indices and bounded
  total bytes. Natural completed tool calls, truncation, cancellation, and failure
  retain distinct terminal meanings. Usage comes from authoritative engine counts.
- Completion usage counts accepted generated tokens, including EOS and tokens
  hidden by string stops. It is not inferred from decoded bytes or emitted text.
  A host string stop awaits execution-owner acknowledgement before reporting final
  usage; pending native work stays owned and cannot accept later output. Terminal
  snapshots retain their counts after owner teardown and output draining.
- Nonstream assembly consumes the same semantic publications and terminal rules
  as SSE. It bounds retained content and final encoded bytes, preserves reasoning
  and append-only tool arguments, and publishes only a terminal response with
  authoritative usage. Failure and cancellation do not become partial successes.

## Host boundary and caches

HTTP connections own their pending handler and response stream, including the
chat session; a detached producer must not outlive a disconnected receiver.
Connection count, request bytes, body/header wait time, and response bytes are
bounded by host configuration. Invalid wire data and unsupported policies remain
distinct client errors; worker unavailability remains a service error. Health
reports actual execution-owner availability and configured model metadata.
Shutdown releases connections before joining the execution owner, which retains
submitted native work until completion.

Immutable tokens, prepared media, options, and symbolic constraint plans cross into
the execution owner. Tokens and snapshots cross out. Live parsers, processors,
matchers, sequences, and device objects retain their owning context.

Template, profile, and grammar caches are bounded and keyed by relevant source,
options, tokenizer, artifact, and implementation identity. Preparation and parsing
measurements remain distinct from engine service and native execution timings.
