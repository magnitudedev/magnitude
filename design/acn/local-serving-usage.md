---
applies_to:
  - packages/acn/src/serving-usage*
  - packages/acn/src/server.ts
  - packages/acn-protocol/src/schemas/serving-usage.ts
  - packages/acn-protocol/src/boundary/models.ts
  - packages/client-common/src/operations/models.ts
  - desktop/src/serving-usage*
---

# Local serving usage

ACN retains local inference usage observations independently of desktop windows, connections,
model residency and process restarts. The ledger contains model identity, completion time, native
token counts and timing evidence; it never retains prompts, generated content or credentials.
It covers local generation through the public service's Chat Completions, Messages and Responses
HTTP/SSE routes and local Codex Responses WebSockets. Cloud forwarding, discovery, token counting,
rejected requests and WebSocket warm-up are excluded.

A passive observer at the private inference forwarding boundary preserves request and response
bytes, headers, cancellation and backpressure. It consumes the generated native usage contracts
and an established SSE parser. It never alters inference requests to obtain more statistics, derives
tokens from text, or creates another native client or inference lifecycle. Each HTTP response and
each accepted WebSocket generation has one observation identity and is committed at most once.
The ledger uses transactional local storage; reads and writes serialize and successful records
survive restart. Storage failure cannot stop inference or masquerade as empty history.

Input includes cached input. Total is input plus output; cached input is never added twice.
Terminal native timings take precedence over boundary measurements. When native timing is absent,
streaming latency measures request start to first output and generation time measures first output
to terminal evidence. Non-streaming responses without native timings contribute no timing sample.
Generation speed is total output over total measured generation time, not the mean of request
rates. First-token latency is the mean of measured samples. Missing values are excluded rather than
zero-filled. Cancelled or truncated responses without terminal counters remain explicitly incomplete.
Messages streaming currently lacks measured cache reads; this is unavailable evidence, not zero.

The canonical replay-safe model query exposes totals, measurement coverage, known models and storage
availability. Today uses the caller's named time zone and its actual local midnight, including DST;
All time covers retained history. Filtering changes presentation, never storage. Client-common owns
observed query refresh so midnight rollover and newly completed requests become visible without
component polling or a second cache. An unobserved window does not acquire a polling lifetime.

Acceptance requires real serving counters to agree with the native response, persistence after
restart, isolated model/date views, missing-evidence presentation, cancelled stream cleanup,
byte-for-byte forwarding and exclusion of cloud traffic. Renderer counters or telemetry traces
cannot substitute for durable accounting.
