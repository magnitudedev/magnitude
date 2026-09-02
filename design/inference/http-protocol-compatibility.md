---
applies_to:
  - inference/crates/icn-contracts/src/inference/**
  - inference/crates/icn-engine/**
  - inference/crates/icn-api/**
  - inference/crates/icn-server/src/inference_worker.rs
  - packages/icn-protocol/**
  - packages/openapi-effect/**
  - packages/acn/src/inference-gateway.ts
  - packages/acn/src/server.ts
  - packages/sdk/src/inference-client.ts
  - cli/src/harness-connections/**
---

# Inference HTTP protocol compatibility

ICN exposes one local inference capability through independent OpenAI Chat Completions, OpenAI
Responses, and Anthropic Messages adapters. Protocol DTOs and streaming state machines remain at
the HTTP boundary; the runner and engine consume only the canonical inference contract.

## Canonical input context

`InferenceContext` contains one optional leading `system` prompt and a nonempty ordered sequence of
user and assistant entries. A system prompt cannot occur among conversation entries. Protocol
inputs containing a system or developer instruction after conversation begins are rejected. The
one exception is Anthropic `system` role messages, which are reinterpreted as positioned user
entries rather than system prompts (see the adapter notes below).

A user entry contains zero or more ordered text/image values. An assistant entry has fixed optional
reasoning and text fields plus zero or more ordered tool exchanges.
Each tool exchange pairs one tool call directly with exactly one result; incomplete or dangling
historical tool interactions cannot enter the context.

Canonical text has exactly two states: absent or nonempty. Optional reasoning, system, and text
fields therefore use optional nonempty text; content collections and tool-call collections are
ordinary vectors and may be empty. Empty text fragments never enter the canonical contract.
Adapters map omitted, null, empty-string, and empty-array forms only where their protocol permits
those forms, preserving protocol-native validation elsewhere. Empty user messages, assistant
messages, and tool results remain meaningful empty containers rather than acquiring placeholder
text. Protocol projection maps absent text and empty collections to that protocol's native wire
representation without inventing semantic output.

The Responses adapter accepts both explicit `message` input items and the standard easy-message
form whose `type` discriminator is omitted. Stateless replay accepts output-message metadata and
reasoning items emitted by the Responses stream. A reasoning item immediately preceding assistant
text or function calls belongs to that same canonical assistant entry; a standalone reasoning item
remains a reasoning-only assistant entry. Request hints such as `prompt_cache_key` may be accepted
without changing canonical inference semantics.

External compatibility request objects tolerate unknown members recursively. Local adapters
discard those members; known fields retain their type and value validation, and tagged protocol
objects retain strict discriminator values. Arbitrary map-valued protocol fields remain data and
follow their field-specific handling. Unsupported semantics represented by known fields still fail
before generation rather than being silently reinterpreted.

Responses tolerance follows one partition rather than per-field patches. Function tool
declarations are the executable semantic core: their required fields and discriminator are parsed
strictly while additional members are ignored. Every other tool declaration — namespace,
web-search, and any future hosted type — is opaque by policy: retained verbatim for protocol
projection, never locally executable, and requiring no adapter change when new hosted types appear.
Replayed items obey a closure invariant: every item the Responses projection can emit parses back
as input, because clients resend emitted output (with its item IDs, statuses, and annotations)
verbatim as later history.

The Chat Completions adapter accepts standard client metadata that does not alter local inference.
In particular, `store: false` and a function definition's optional `strict` flag are valid input.
`store: true` is rejected explicitly because the local runtime does not provide server-side
conversation persistence; it is not rejected as an unknown field.
Historical Chat Completions tool-result messages may include the function name in addition to the
required tool-call ID. Anthropic Messages accepts cache-control annotations and current Claude Code
output configuration. Unknown output-configuration members remain compatibility hints, while
`output_config.effort` is a semantic input owned by the Anthropic adapter.

Reasoning-effort compatibility is one unconditional local-admission invariant. Every Chat
Completions, Responses, and Anthropic request applies `RoundUpOrClamp` after model resolution.
Requests carry no trusted harness identity, so there is no adapter field or selectable resolution
mode. This is scoped reasoning-effort compatibility, not generic request rewriting.

An effort that exactly names a supported model behavior remains unchanged. For the ordinal scale
`minimal < low < medium < high < xhigh < max`, an unsupported value selects the least supported
enabled ordinal at or above it, or the greatest supported enabled ordinal below it when no higher
value exists. An unsupported named behavior selects the enabled model default. Omitted or
`adaptive` effort selects the model default. Disabled reasoning remains valid only when the model
supports `none`.

Admission applies reconciliation only after leasing exposes the model profile. The behavior is
neither a canonical reasoning intent nor a serializable provider-model value. Connectors must still
project the most precise native effort domain and default each harness can express; global safety is
not a substitute for accurate harness controls.

Reasoning-history controls are independent from current-turn reasoning effort. Chat Completions
passes explicitly supplied history controls through to the effective template unchanged.

The local Anthropic adapter also accepts text-only `system` role messages. The current Anthropic
protocol permits system-role messages mid-conversation as an operator channel that leaves the
cached prompt prefix untouched, and Claude Code uses it to surface text the user types mid-turn.
Local chat templates have no reliable mid-sequence system turn and canonical context carries
exactly one leading system prompt, so the adapter reinterprets each system-role message as a user
entry at its original position. This is Magnitude's deliberate local interpretation: conversation
order and the local prompt prefix are preserved, and mid-stream operator authority is not claimed.
Because the mapping is user-authority, a system-role message is accepted anywhere a user message
is legal — intentionally more lenient than the upstream protocol's placement rules, which exist to
protect cache and authority semantics this mapping does not carry.

The Anthropic adapter also owns the Claude Code attribution projection. Claude Code prepends
provider-reserved billing metadata as the leading system text
(`x-anthropic-billing-header: …; cch=<stamp>;<optional real prompt>`), which api.anthropic.com
strips before the model sees it; the per-request `cch` stamp would otherwise defeat prompt-prefix
reuse and the metadata would become model-visible prompt content. The projection runs on the
leading system text before blocks are joined, so Messages and token counting share it through the
common adaptation path. Recognized attribution is removed and any real prompt suffix is preserved
exactly; all other content passes through byte-identical; an unrecognized sentinel shape is
preserved with a content-free diagnostic rather than guessed at. The projection is idempotent and
exists only in this adapter: the byte-preserving ACN gateway and the canonical inference domain
never learn this metadata exists. Magnitude-launched Claude Code additionally sets
`CLAUDE_CODE_ATTRIBUTION_HEADER=0`, so the projection covers only clients Magnitude did not
launch. The durable rule: provider-reserved attribution is protocol metadata, not prompt content,
and it dies at the protocol boundary.

Input context and generated output are separate domains. A context entry is never a generation
result, and inference output is never a context entry. The inference contract does not own how a
caller constructs a later invocation from an earlier result.

## Execution and output

Protocol adapters construct a validated `InferenceInvocation` directly. Its
`InferenceModelSelector` is a nonempty public selection that may name a controller alias; the
shared admission runner resolves it to one exact model lease. This is intentionally distinct from
the inventory `ModelId`, whose syntax identifies only canonical installed artifacts. The resident
backend accepts only a model-resolved request. Model-native message and template construction are
private to the engine's template compiler.

An output-token limit is caller policy, not an adapter default. Chat Completions carries
`max_completion_tokens` or `max_tokens` when supplied, Responses carries `max_output_tokens` when
supplied, and omission remains absence in the canonical request. The engine then permits generation
through the model's remaining context capacity. Anthropic Messages is unchanged: its protocol
requires `max_tokens`, so the adapter always supplies that explicit limit. No adapter invents a
finite limit for a request that omitted one.

Prompt capacity is independent of that output policy. After model acquisition, the resident worker
prepares the native prompt exactly once and reports admission before semantic output. A prompt at
or above the configured per-request context capacity cannot leave room for generation. Ordinary
Chat Completions, Responses, and Anthropic Messages requests await this admission before opening a
successful response or event stream, so the failure is a protocol-native HTTP error. Chat
Completions and Responses use an OpenAI `invalid_request_error` with code
`context_length_exceeded` and request-field `param`; Anthropic Messages uses its
`invalid_request_error` and `prompt is too long` message. Responses WebSocket necessarily reports
the error through its upgraded protocol. Byte-preserved upstream requests remain excluded.

`Magnitude-Include-Progress: true` deliberately changes the HTTP commitment point for streaming
Chat Completions and Responses. The endpoint opens SSE before model acquisition and forwards
loading fractions live as they occur. A later load, admission, or execution failure therefore uses
the protocol's native in-stream failure. Progress is never buffered for replay. Text and multimodal
inputs use the same authoritative resident preparation path; no estimate may produce a
context-length rejection.

The engine emits fixed reasoning, text, and tool-call output phases plus terminal usage,
termination, and metrics facts. It does not construct an aggregate output. Lifecycle progress is
observation, not semantic output. One bounded output journal validates the event lifecycle,
constructs the aggregate output exactly once, and supplies the same result to streaming and
non-streaming projections.

Chat Completions and Responses are single mixed HTTP operations: `stream=false` returns their JSON
response and `stream=true` returns their protocol event stream. OpenAPI declares both media types,
and the generated client derives both calls from that one operation. SDK code must not recreate an
untyped HTTP path beside the generated transport.

Anthropic token counting runs the same canonical request conversion and model-native prompt
preparation as Messages generation on a resident model, then returns the exact logical input-token
count.
Streaming Messages reports that count in `message_start` before output blocks.

## Public routing

ACN exposes generic local OpenAI traffic under `/inference/v1/**` and generic local Anthropic
traffic under `/inference/anthropic/**`. These generic surfaces do not expose harness-specific
aliases or discovery. Codex multiplexing is isolated under `/inference/v1/proxies/codex/**` with
the reserved `magnitude-local/<canonical-model-id>` namespace. Claude Code multiplexing is isolated
under `/inference/anthropic/proxies/claude-code/**` with the reserved
`anthropic-local/<canonical-model-id>` namespace.

The OpenAI model-list response is one standard `object: "list"` envelope with an enriched `data`
array. Each model entry retains the standard identity fields and follows the OpenRouter discovery
conventions for `name`, `description`, `context_length`, `architecture`,
`supported_parameters`, `top_provider`, and optional `reasoning`. `reasoning` publishes the exact
normalized effort domain and default plus whether the default is enabled and whether reasoning is
mandatory. `top_provider.max_completion_tokens` is the connector-facing output ceiling, bounded
to 32,768 tokens and never greater than the configured context window. There is no parallel
Magnitude-owned descriptor array; SDK, ACN, CLI, and external harnesses consume the same rows.
This extension is discovery metadata only and does not change model selection semantics.
Runnable external Hugging Face models appear in the array under the exact canonical
`hf:<owner>/<repository>/<artifact-selector>` identity. Path transport percent-encodes that whole
identity where it occupies one URL segment; JSON bodies and model-list values do not encode or
rewrite it.

For a reserved harness-proxy alias, ACN removes any caller-supplied alias-echo header, rewrites only the request
model to its canonical ID, removes caller credentials, installs private ICN authorization, and
sets the validated alias for response model echoing. The alias never enters the canonical
invocation, model controller, worker transport, or engine.

For non-reserved Claude Code models, ACN forwards the original request bytes and Anthropic headers
upstream. For non-reserved Codex models, ACN selects the fixed OpenAI API or ChatGPT Codex origin
from Codex's request headers and forwards the request and credentials. Codex response items minted
by ICN are the sole request-body exception: local reasoning items are omitted because they have no
portable encrypted payload, local message IDs are removed so their content remains inline, and a
local previous-response ID is removed. OpenAI must never receive an ICN ID as a reference to
OpenAI-hosted state. Upstream traffic never enters ICN, and local aliases never reach upstream.

The harness gateways are deliberately routing shims rather than protocol adapters. They recognize
only gateway-owned paths and the one top-level model discriminator needed to choose a target. They
do not define provider request DTOs, validate message or tool content, parse response bodies or
SSE events, or normalize evolving vendor fields. Classification is size-bounded. A local request is
changed mechanically only at the model string; every other JSON field remains opaque. An upstream
request retains its original bytes, query, public headers, response status, headers, and body
stream. Except when removing ICN-owned history references, Codex zstd request compression is
decoded only into a bounded routing copy and the original compressed bytes go upstream unchanged.
Proxy fetches disable automatic response decompression so
an upstream `Content-Encoding` header always describes the bytes returned to the harness.
Codex's proxy provider retains the semantic provider name `OpenAI` and native OpenAI authentication
and declares native Responses WebSocket support. ACN classifies every `response.create` frame and
maintains one active outbound socket for that frame's destination, replacing it when the selected
model crosses the local/upstream boundary. Server frames are relayed opaquely. ICN implements the
local Responses WebSocket state machine, including connection-scoped previous-response history;
ACN does not translate WebSocket events into SSE.

Claude Code receives the local base URL and gateway discovery switch through its user-wide settings.
Magnitude writes no persistent Claude credential. Ordinary Claude models retain the user's
authentication and pass through the gateway to Anthropic, while discovered
`anthropic-local/<canonical-model-id>` aliases route to ICN. When connection receives an explicit
model, it persists that alias as Claude Code's ordinary startup model; a launch plan independently
passes the same explicit alias.
The discovery response uses the model's ordinary description and does not encode capability
metadata as prose. Claude Code's current gateway schema cannot consume a machine-readable effort
domain, so effort interpretation remains in the local Anthropic adapter after model resolution;
ACN continues to rewrite only the routing alias.

## Ownership

- ICN contracts own input, execution, output, usage, termination, and failure facts.
- ICN API adapters own DTO validation, request conversion, protocol IDs, wire event ordering, and
  protocol-specific errors.
- The shared runner owns exact model leasing, global reasoning-effort reconciliation against the
  resolved model, reasoning resolution, execution, cancellation, and bounded output delivery. Wire
  compatibility never extends the canonical inference contract.
- ACN owns harness-proxy model routing, local aliases, Claude Code discovery projection, credential
  isolation, and byte-preserving upstream proxying. Request mutation is limited to replacing a
  local routing alias and removing ICN-owned response references before an upstream Codex request.

No adapter converts through another adapter's DTO. No generic adapter abstraction may erase the
distinct protocol state machines.

## Conformance

- Equivalent requests in every supported dialect construct equal canonical invocation values.
- Unsupported semantics fail before generation rather than being silently reinterpreted.
- Streaming events reconstruct the same semantic output and terminal result as non-streaming mode.
- Empty text and tool-input deltas do not create canonical output or protocol output items/blocks.
- Permitted empty wire content round-trips through empty canonical containers without placeholder
  strings; protocol-forbidden empty forms remain request errors.
- Every historical tool call in input context has exactly one result in the same assistant entry.
- OpenAI and Anthropic errors retain their native HTTP and in-stream envelopes.
- For ordinary HTTP requests, a local prompt that leaves no generation capacity fails with a
  protocol-native HTTP error before an HTTP streaming success response is opened. Explicit
  Magnitude progress streams report failures discovered after opening in-stream.
- Generated mixed-operation clients validate JSON success bodies and promote declared HTTP or SSE
  errors through one typed runtime path.
- Client cancellation releases the exact inference lease and emits no further client output.
- Upstream Anthropic traffic never enters ICN and local aliases never reach an upstream provider.
- Caller-supplied private authorization and alias-echo headers cannot cross the public boundary.
