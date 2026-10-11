---
applies_to:
  - inference/engine/serving/**
  - inference/engine/cli/**
  - inference/service/server/src/serving.rs
  - inference/engine/src/chat/**
  - inference/engine/src/composition.rs
  - inference/engine/src/invocation.rs
  - inference/engine/chat/src/request.rs
  - inference/engine/chat/src/output.rs
  - inference/engine/chat/src/stream.rs
  - inference/engine/chat/src/generation.rs
  - inference/engine/generation/src/controls.rs
---

# Inference serving

The engine's protocol library is the one implementation of Chat Completions,
Responses (HTTP, SSE and WebSocket) and Anthropic Messages with `count_tokens`.
The service mounts it beside its management routes; the standalone engine
binary mounts it for its one explicit model. Handlers never load models or
choose catalog entries: they name a model to a model source and receive either
a generation binding (which may load the model and holds its release guard) or
its host chat semantics (which never lease or load). A binding is to exactly
one model generation.

Every protocol adapts into one protocol-neutral request: a conversation (one
leading system prompt and ordered user and assistant entries with complete tool
exchanges), offered tools with their selection and parallelism, a reasoning
intent, an output format, caller template arguments and generation controls.
Rendering that request is the only path to template input, so counting,
template application and generation see identical prompts and counts agree
with generation. JSON mode constrains output to a JSON object; a caller grammar
constrains output to its language; tool and output JSON schemas constrain output
best effort, as the schema constraints design defines. Enforcement accepts exactly
the grammar's language and admits end of generation only where it accepts. Image sources are validated
and decoded by the protocol layer (base64 data URLs only, bounded); network URLs
are never fetched. The standalone binary additionally reports readiness identity
(`/health`: model, served context, vocabulary) and exposes counting at
`/v1/count` for benchmark preparation.

The host owns request validation, prompt rendering, tokenization, reasoning
resolution, constraint description, text/tool parsing, protocol framing and
transport lifetime. The worker builds the engine blueprint and owns model
execution, state, scheduling, sampling and token constraints. Rendered tokens,
generation options and a constraint description cross the worker boundary; live
tokenizers, parsers and device objects do not.

The host resolves one explicit served-context bound within the artifact's declared capability.
That bound is part of the resolved model definition shared by input preparation, numerical planning,
state allocation, readiness, and request validation. A prompt must leave at least one served
position for generation. Counting is a host-only sizing operation: it renders and counts inputs up to
the artifact capability through a separate input-preparation authority, but it neither enlarges the
served bound nor allocates numerical state at the counted size.

Tool selection has one meaning in both the prompt and the output language. An
automatic selection offers the supplied tools. A named selection offers the
selected function; a required selection tells the model to call an available
function; an allowed-subset selection offers only that subset, optionally
required. A selection of none omits tools. Rendering communicates required/named
selection without mutating the caller's history; exact instruction wording is
not a performance guarantee. The checkpoint still owns chat and argument syntax.
Prompt instructions communicate intent; request-local grammar enforces it.
Under a selection that requires a call, the output language admits only
reasoning before it, never content: content could not end, since the turn
cannot end without the call. Reasoning is read the way the output parser reads
it, so reasoning the prompt opens closes before the turn ends.
Template keyword overrides cannot replace request-owned tool selection or other
rendering inputs.

Reasoning requests resolve against the model's detected profile: an exact
effort is preserved; an unsupported ordinal effort rounds up to the nearest
declared enabled level and clamps to the highest; a named mode without a rank
falls back to the enabled default; disabling a model that cannot disable
reasoning is an error. The model default renders exactly as the template's own
default. A reasoning budget is a hard engine cap: once the budget of reasoning
tokens is spent without the model closing reasoning, selection is restricted to
the template's reasoning end tag. No budget is ever derived from an effort.
Ignoring end of generation removes the model's stop tokens from every selection.
Prompt caching allowed retains exact prompt prefixes; disallowed requests are
transient.

Stream parsing emits semantic text, reasoning and complete validated tool calls
in one ordered stream that non-streaming responses also assemble from. An empty
generation is an empty assistant turn in each protocol's native shape, and an
empty assistant turn in history (that output replayed, or a client's record of
a step that failed before any output) contributes nothing to the prompt. Each
tool call's ID is unique across responses, since clients key calls and results
by ID over a whole conversation: the ID the model wrote when its format carries
one (templates render it back), otherwise a fresh random one. A
caller stop sequence ends output with a stop-sequence termination; tool calls
terminate as tool calls; the output limit or context end is a length
termination. Output that stops inside a tool call (at the output limit, the
context end or a stop sequence) terminates as that stop, never as tool calls and
never as a failure: the cut-off call is incomplete, so complete responses hold
only the calls that completed. A stream has already published the incomplete
call's partial input; Responses concludes it as an `incomplete` item outside the
response's `output`. A request samples with the seed it names (every
protocol accepts `seed`) or, when it names none, a fresh one, so retrying an
identical request samples anew; a seed's draws are position-addressed, so
batching and speculation never change them. Terminal usage and timings come from actual engine execution;
progressive per-token timings are host-observed. Client disconnection cancels
its request; one request's cancellation does not retire shared device work
still used by peers.

An ordinary stream commits its HTTP response only after the engine admitted the
request, so preparation and admission failures keep their HTTP status. A
progress stream (`Magnitude-Include-Progress`) commits immediately, reports
model loading (its stage and fraction, named as the service's Instance status names
them) and request stages (preparing, queued, prefill, generating; prefill
reports completed, total and prefix-reused prompt tokens), and
reports every later failure in-stream. Engine outcomes keep their typed meaning
on the wire: memory, observation and capacity refusals are retryable 503s, an
unsupported model is a non-retryable conflict, and device loss is a retryable
server error. A failure of an admitted request is an engine failure, never a
client error.

The Responses WebSocket keeps per-connection logical history in memory only:
the most recent concluded (completed or incomplete) response, as its full input
followed by its output items in `output` order. A request marked not to generate
is validated as its generation would be and recorded without generation. A
later request naming that response as its predecessor continues from it and
appends its own input; naming any other is `previous_response_not_found`, and a
continuation that fails evicts the predecessor it named. Output items conclude
in `output_index` order, so the order a client observes them is the order of
`output`. A request that fails before it becomes a response is answered with an
`error` event carrying the HTTP status it would have had and the standard error
object, so the client ends the request rather than waiting for a terminal
response event.

Qualification covers immutable input history, automatic/required/named/allowed/disabled
tools, reasoning resolution and budgets, grammar and schema enforcement,
output cut off inside a tool call in streaming and complete responses of every
protocol, fragmented streams, backpressure, cancellation, replay closure of every emitted
output shape, and consistency of terminal timing and token counts between
counting and generation.
