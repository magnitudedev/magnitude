---
applies_to:
  - packages/ai/src/codec/native-chat-completions/**
  - packages/ai/src/protocol/native-chat-completions.ts
  - packages/ai/src/wire/chat-completions/**
  - packages/ai/src/model/define.ts
  - packages/ai/src/options/option.ts
  - packages/ai/src/transport/stream.ts
  - packages/providers/src/custom-endpoint/**
  - packages/providers/src/magnitude/**
  - packages/icn/src/provider/**
  - packages/inference-benchmark/**
---

# Native Chat Completions

## Contract ownership

The native Chat Completions protocol is the shared provider-agnostic contract for OpenAI-compatible
chat requests. Its request and prompt contribution Schemas own the distinction between decoded
domain values and encoded JSON. Its protocol builder owns request assembly policy independently of
HTTP, ICN, or any other transport.

Codecs translate provider-neutral prompts into Schema-encoded prompt contributions. Option
definitions translate call options into Schema-encoded request contributions. The native protocol
combines those contributions with protocol constants and optional provider composition.
`Model.define` owns the final protocol-Schema encode immediately before generic transport and
tracing.

## Optional values and extensions

Every optional decoded request value is an exact `Option`. `Option.none` means the encoded property
does not exist; `undefined` and `null` are not request-side absence values. The encoded request is a
JSON object with precise standard properties and may contain provider-specific JSON extensions.

Extensions are explicit in the decoded request and flat in the encoded request. They cannot own a
standard protocol property. Independently produced request contributions are merged with duplicate
property detection rather than last-write-wins assignment.

## Assistant history

Every emitted assistant request message has string-valued `content`. When semantic history has no
visible assistant text, the portable encoded representation is the empty string. Reasoning and tool
calls remain independently optional and are omitted when absent. A wholly empty semantic assistant
turn remains representable as an assistant message with `content: ""`; request encoding does not
delete or reinterpret semantic history.

## Construction and failure behavior

Hosted, custom-endpoint, and ICN requests use the same native request builder. The builder:

1. Schema-encodes each supplied option contribution;
2. Schema-encodes the prompt contribution;
3. collision-checks all contributions;
4. decodes the complete request to its Option-native value;
5. preserves omission of caller-owned optional fields; and
6. invokes optional provider composition within the typed Effect failure channel.

Mapper throws, composition throws, contribution encoding failures, collisions, and final request
encoding failures are typed client-correctness failures. They do not escape as Effect defects and
no transport attempt occurs.

## Transport adapters

Generic HTTP transport receives only the final encoded JSON request. Traces retain that exact
protocol-specific encoded type and value.

ICN runs the shared native builder, encodes the native request, adds only ICN-owned addressing, and
then validates the result through the generated ICN request contract. The inference benchmark is a
direct wire client: it may add benchmark-owned JSON extensions, but validates and canonicalizes the
complete request through the native request Schema before fetch.

## Responses

Response fields that are nullable on provider wires accept either missing or `null` and decode both
states to `Option.none`. Optional non-null extension fields preserve their existing contracts.
Downstream response normalization does not carry wire-level nullability.

## Acceptance criteria

- No emitted assistant request has null or missing `content`.
- No optional request property is encoded as `undefined` or `null`.
- All option and prompt contributions are Schema-encoded before merging.
- No contribution or extension can silently replace another owner's property.
- Hosted, custom-endpoint, and ICN request construction preserve caller omission and share failure behavior.
- Provider-specific JSON extensions survive final encoding without weakening standard fields.
- Composition cannot bypass request validation or escape the typed failure contract.
- Nullable response fields preserve missing/null wire acceptance and expose decoded absence as `Option`.
