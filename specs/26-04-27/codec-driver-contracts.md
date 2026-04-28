# Codec / Driver Contracts

**Status:** Draft
**Date:** 2026-04-27

## Purpose

Establish the contracts that let us implement the **native** paradigm (OpenAI-compatible chat completions with structured `tool_calls` + `reasoning_content`) end-to-end, with a clear path forward for the **completions** paradigm (raw text in each model family's native trained format, parsed via per-family adapters) and the **xml-act** paradigm (existing universal text-tag parser, eventually ported to fit these contracts).

This spec focuses on **interfaces and data shapes**, not implementations. Once these contracts are locked in, the native paradigm is implemented by writing one Driver, one Codec, and one Model record. Completions and xml-act follow the same pattern.

## Design Principles

1. **Composition over dispatch.** No string-literal unions selecting codec / driver / model family. A `Model` holds **instances** of `Codec` and `Driver`. Adding a new paradigm or provider means constructing a new instance, not editing a switch.

2. **Single canonical memory form.** All paradigms read from and write to the same `Message[]` shape per fork. Wire-format differences are absorbed by Codecs at encode/decode time. No paradigm-specific branches in the storage layer.

3. **Streaming-first.** Codec decode produces a `Stream<TurnPartEvent>`. Memory and UI both consume that stream; nothing waits for a turn to "finish" before observing it.

4. **Effect-native.** All asynchronous operations return `Effect.Effect<...>`. Streams use `Stream.Stream<...>`. Services use `Context.Tag` (matching codebase convention; not `ServiceMap.Service`). Schemas use `Schema.Class` and `Schema.TaggedClass`.

5. **Multimodal as first-class content.** `ContentPart` (text + image) is the lowest common denominator across all paradigms. Empirically verified that multimodal tool messages work on Fireworks/Kimi K2.6 chat completions. Codecs may still need fallback strategies for stricter providers; that's a codec implementation detail, not a storage concern.

6. **IDs are ours.** We generate `ToolCallId` values during parsing/encoding and ignore server-provided IDs. Verified compatible with Fireworks. This means a single ID space spans memory, codec events, and the wire.

---

## Table of Contents

1. [Glossary](#1-glossary)
2. [Architecture Overview](#2-architecture-overview)
3. [Package Layout & Dependencies](#3-package-layout--dependencies)
4. [Branded IDs](#4-branded-ids)
5. [Content & Tool Definitions](#5-content--tool-definitions)
6. [Memory Shape (Storage Form)](#6-memory-shape-storage-form)
7. [TurnPart (Assistant Turn Output)](#7-turnpart-assistant-turn-output)
8. [ResultEntry & TimelineEntry (Inbox Contents)](#8-resultentry--timelineentry-inbox-contents)
9. [TurnPartEvent (Streaming Decode Events)](#9-turnpartevent-streaming-decode-events)
10. [Wire Types](#10-wire-types)
11. [Codec Interface](#11-codec-interface)
12. [Driver Interface](#12-driver-interface)
13. [ModelAdapter Interface](#13-modeladapter-interface)
14. [Model Record (Composition Node)](#14-model-record-composition-node)
15. [TurnEngine (Orchestrator)](#15-turnengine-orchestrator)
16. [Effect Services](#16-effect-services)
17. [Lifecycle: Encoding a Turn](#17-lifecycle-encoding-a-turn)
18. [Lifecycle: Decoding a Turn](#18-lifecycle-decoding-a-turn)
19. [Lifecycle: Tool Execution](#19-lifecycle-tool-execution)
20. [Paradigm: Native (Chat Completions)](#20-paradigm-native-chat-completions)
21. [Paradigm: Completions (Raw Text + Adapter)](#21-paradigm-completions-raw-text--adapter)
22. [Paradigm: xml-act (Universal Text Tags)](#22-paradigm-xml-act-universal-text-tags)
23. [ID Generation & Pairing](#23-id-generation--pairing)
24. [Multimodal Handling per Paradigm](#24-multimodal-handling-per-paradigm)
25. [Error Handling](#25-error-handling)
26. [Tracing](#26-tracing)
27. [Implementation Plan](#27-implementation-plan)
28. [Open Questions](#28-open-questions)
29. [Appendix: Mapping to Existing Code](#29-appendix-mapping-to-existing-code)

---

## 1. Glossary

| Term | Meaning |
|---|---|
| **Memory** | The persistent, fork-scoped store of conversation state. A `Message[]` per fork. Canonical, paradigm-agnostic. |
| **Message** | One element of memory. A tagged record (e.g. `assistant_turn`, `inbox`, `session_context`). |
| **TurnPart** | One structured output unit produced by the model during an assistant turn (a thought, a tool call). |
| **TurnPartEvent** | Streaming event emitted by `Codec.decode` while the model generates. Granular: `ThoughtStart`, `ThoughtDelta`, `ThoughtEnd`, `ToolCallStart`, etc. |
| **Inbox** | A `Message` containing everything between assistant turns: tool results, user messages, subagent activity, observations, etc. Two parts: `results: ResultEntry[]` and `timeline: TimelineEntry[]`. |
| **ResultEntry** | An inbox entry that is a *response* to something (tool result, parse error, message ack, etc.). |
| **TimelineEntry** | An inbox entry that is *new context* (user message, subagent block, presence, observation, task update, etc.). |
| **ContentPart** | The renderable content primitive: `text` or `image`. Multimodal-aware. Reused from `@magnitudedev/tools`. |
| **ToolDef** | A tool's name, description, and parameter schema. The codec serializes this to the wire format the model expects. |
| **Wire types** | Typed representations of HTTP request and streaming-chunk shapes for a given API surface (e.g., chat completions, raw completions). |
| **Codec** | The mapping between Memory + ToolDefs and the wire (encode), and between the wire stream and `TurnPartEvent`s (decode). One per paradigm. |
| **Driver** | Pure transport: send a typed request, return a typed stream of chunks. One per API surface. |
| **ModelAdapter** | Per-family encoder/decoder for the completions paradigm. Knows how a specific model family (Kimi, GLM, Qwen, etc.) tokenizes its assistant turns. Used internally by `CompletionsCodec`. |
| **Model** | The composition node: an `id`, capabilities, a Driver instance, and a Codec instance. |
| **TurnEngine** | The orchestrator service that runs the encode → driver → decode → tool-exec → repeat loop. |
| **Paradigm** | One of: `native`, `completions`, `xml-act`. A combination of Driver and Codec strategy. |
| **Fork** | A branch of a session's conversation. Memory is per-fork. |

---

## 2. Architecture Overview

```
                         ┌──────────────────────────────────┐
                         │            TurnEngine             │
                         │   (Context.Tag service)           │
                         │   reads Memory, drives loop       │
                         └──────────────┬────────────────────┘
                                        │
                         ┌──────────────▼─────────────┐
                         │        Model record         │
                         │   { id, capabilities,       │
                         │     driver: Driver,         │
                         │     codec: Codec }          │
                         └──────┬───────────────┬──────┘
                                │               │
                    ┌───────────▼──┐    ┌───────▼──────┐
                    │    Codec     │    │    Driver    │
                    │  (paradigm)  │    │  (transport) │
                    │              │    │              │
                    │  encode:     │───▶│   send:      │
                    │   Memory →   │    │   Wire req → │
                    │   Wire req   │    │   Wire stream│
                    │              │◀───│              │
                    │  decode:     │    └──────────────┘
                    │   Wire stream│
                    │     →        │
                    │   Stream of  │
                    │   TurnPart   │
                    │   Events     │
                    └──────────────┘
```

**Concrete combinations:**

| Paradigm | Codec | Driver | Notes |
|---|---|---|---|
| **native** | `NativeChatCompletionsCodec` | `OpenAIChatCompletionsDriver` | Structured `tool_calls`, `reasoning_content`. Multimodal in tool messages OK on Fireworks. |
| **completions** | `CompletionsCodec(adapter)` | `OpenAICompletionsDriver` | Raw text completions endpoint. Codec is parameterized by a `ModelAdapter` for the model family. |
| **xml-act** | `XmlActCodec` | `OpenAIChatCompletionsDriver` | Renders all messages as text; parses tags out of the assistant text stream. |

The **same** `OpenAIChatCompletionsDriver` is used by both `native` and `xml-act`. Differences are entirely in the codec.

---

## 3. Package Layout & Dependencies

```
packages/codecs/
  src/
    memory/
      message.ts          # Message Schema.Union of TaggedClasses
      turn-part.ts        # TurnPart Schema.Union
      content-part.ts     # re-export from tools
      result-entry.ts     # ResultEntry Schema.Union
      timeline-entry.ts   # TimelineEntry Schema.Union
      ids.ts              # branded IDs (TurnId, ToolCallId, ...)
    events/
      turn-part-event.ts  # TurnPartEvent Schema.Union
    tools/
      tool-def.ts         # ToolDef Schema.Class
    codec.ts              # Codec interface
    adapters/
      model-adapter.ts    # ModelAdapter interface
      kimi.ts
      glm.ts
      qwen.ts
      ...
    impls/
      native.ts           # NativeChatCompletionsCodec
      xml-act.ts          # (later) XmlActCodec
      completions.ts      # CompletionsCodec(adapter)
    index.ts
  package.json

packages/drivers/
  src/
    driver.ts             # Driver interface
    wire/
      chat-completions.ts # ChatCompletionsRequest, *StreamChunk
      completions.ts      # CompletionsRequest, *StreamChunk
    impls/
      openai-chat-completions.ts
      openai-completions.ts
    index.ts
  package.json
```

**Dependency graph:**

```
agent ─┬─▶ providers ─▶ codecs ─▶ drivers
       │                  │          │
       │                  ▼          ▼
       │                tools     (HttpClient, auth)
       │                  ▲
       └──────────────────┘
```

- `codecs` knows nothing about HTTP. Imports Wire types from `drivers`.
- `drivers` knows nothing about Memory or TurnPart. Pure transport with typed wire shapes.
- `providers` defines the `Model` record, composing a Driver and a Codec.
- `agent` consumes `Model` via `TurnEngine` and drives the loop.

---

## 4. Branded IDs

All IDs are branded primitives via `Schema.brand`. Cross-type confusion is a compile error.

```ts
// packages/codecs/src/memory/ids.ts
import { Schema } from 'effect'

export const ForkId      = Schema.String.pipe(Schema.brand('ForkId'))
export type  ForkId      = typeof ForkId.Type

export const TurnId      = Schema.String.pipe(Schema.brand('TurnId'))
export type  TurnId      = typeof TurnId.Type

export const ToolCallId  = Schema.String.pipe(Schema.brand('ToolCallId'))
export type  ToolCallId  = typeof ToolCallId.Type

export const ThoughtId   = Schema.String.pipe(Schema.brand('ThoughtId'))
export type  ThoughtId   = typeof ThoughtId.Type

export const ModelId     = Schema.String.pipe(Schema.brand('ModelId'))
export type  ModelId     = typeof ModelId.Type
```

`ToolCallId` format: `call-{ord}-{ts36}` (matches existing `ctx.generateId()` from xml-act parser). Generated by us; we ignore server-provided IDs.

---

## 5. Content & Tool Definitions

### 5.1 ContentPart (re-exported from `@magnitudedev/tools`)

```ts
type ContentPart =
  | { readonly type: 'text';  readonly text: string }
  | { readonly type: 'image'; readonly base64: string; readonly mediaType: ImageMediaType; readonly width: number; readonly height: number }
```

This is the lowest common denominator for renderable content. All wire formats we target accept multimodal user content; tool messages on Fireworks accept multimodal content (verified empirically); codec implementations handle stricter providers via fallback (text-only with image moved to follow-up user message).

### 5.2 ToolDef

```ts
// packages/codecs/src/tools/tool-def.ts
export class ToolDef extends Schema.Class<ToolDef>('ToolDef')({
  name:        Schema.String,         // e.g. "read", "shell"
  description: Schema.String,
  parameters:  Schema.Unknown,        // JSON Schema (encoded as Schema.Record(...) in practice)
}) {}
```

Codecs consume `readonly ToolDef[]` to:
- **Native**: emit `tools: [{type:'function', function: {name, description, parameters}}, ...]` in the request body.
- **Completions**: defer to `ModelAdapter.encodeTools(toolDefs)` to produce a system-prompt fragment in the family's expected format.
- **xml-act**: render an XML schema fragment in the system prompt.

---

## 6. Memory Shape (Storage Form)

Memory is `readonly Message[]` per fork. Append-only at the projection layer; ordering is the source of truth for turn boundaries.

```ts
// packages/codecs/src/memory/message.ts
export const Message = Schema.Union([
  SessionContextMessage,
  ForkContextMessage,
  CompactedMessage,
  AssistantTurnMessage,
  InboxMessage,
])
export type Message = typeof Message.Type
```

### 6.1 SessionContextMessage

```ts
class SessionContextMessage extends Schema.TaggedClass<SessionContextMessage>('SessionContextMessage')(
  'session_context',
  {
    content: Schema.Array(ContentPart),
  }
) {}
```

### 6.2 ForkContextMessage

```ts
class ForkContextMessage extends Schema.TaggedClass<ForkContextMessage>('ForkContextMessage')(
  'fork_context',
  {
    content: Schema.Array(ContentPart),
  }
) {}
```

### 6.3 CompactedMessage

```ts
class CompactedMessage extends Schema.TaggedClass<CompactedMessage>('CompactedMessage')(
  'compacted',
  {
    content: Schema.Array(ContentPart),
  }
) {}
```

### 6.4 AssistantTurnMessage

The single most important change from today. Was `content: ContentPart[]` (text blob); now `parts: TurnPart[]` (structured).

```ts
class AssistantTurnMessage extends Schema.TaggedClass<AssistantTurnMessage>('AssistantTurnMessage')(
  'assistant_turn',
  {
    turnId:     TurnId,
    parts:      Schema.Array(TurnPart),    // see §7
    strategyId: Schema.String,             // existing strategy/role identifier
  }
) {}
```

### 6.5 InboxMessage

Unchanged from today's structure. Internal entries gain `toolCallId` where applicable (see §8).

```ts
class InboxMessage extends Schema.TaggedClass<InboxMessage>('InboxMessage')(
  'inbox',
  {
    results:  Schema.Array(ResultEntry),    // see §8.1
    timeline: Schema.Array(TimelineEntry),  // see §8.2
  }
) {}
```

---

## 7. TurnPart (Assistant Turn Output)

A `TurnPart` is one structured unit of model output for a turn. The model's whole turn is `parts: TurnPart[]`.

```ts
// packages/codecs/src/memory/turn-part.ts
export const TurnPart = Schema.Union([
  ThoughtPart,
  ToolCallPart,
])
export type TurnPart = typeof TurnPart.Type
```

> **Note.** "message to user" is **not** a separate variant. Per design changes, message-to-user becomes a tool call (e.g., `send_message`). This makes assistant output uniform: thoughts and tool calls only.

### 7.1 ThoughtPart

```ts
class ThoughtPart extends Schema.TaggedClass<ThoughtPart>('ThoughtPart')(
  'thought',
  {
    id:    ThoughtId,
    level: Schema.Literals(['low', 'medium', 'high']),  // replaces named lenses
    text:  Schema.String,
  }
) {}
```

`level` controls thinking depth/budget instead of the previous named-lens system.

### 7.2 ToolCallPart

```ts
class ToolCallPart extends Schema.TaggedClass<ToolCallPart>('ToolCallPart')(
  'tool_call',
  {
    id:       ToolCallId,
    toolName: Schema.String,
    input:    Schema.Unknown,            // structured arguments (parsed JSON)
    query:    Schema.NullOr(Schema.String),  // optional filter / query (replaces filter system at the data level — but query persists as a per-call concept)
  }
) {}
```

> The previous "filter" system as a streaming construct is dropped. `query` is retained as a tool-call-level field where some tools use it for partial result selection. (Open question §28: drop entirely?)

---

## 8. ResultEntry & TimelineEntry (Inbox Contents)

Inbox structure stays as it is today (`results: ResultEntry[]`, `timeline: TimelineEntry[]`). Two **shape changes** to result entries: tool-relevant entries gain `toolCallId` so codec can pair them with the corresponding `tool_call` in the assistant turn.

### 8.1 ResultEntry (changes)

```ts
class ToolObservationResultEntry extends Schema.TaggedClass(...)('ToolObservation', {
  toolCallId: ToolCallId,                              // ← NEW: required for native pairing
  tagName:    Schema.String,                           // for xml-act rendering & debugging
  query:      Schema.NullOr(Schema.String),
  content:    Schema.Array(ContentPart),               // text + (optional) images, raw tool output
}) {}

class ToolErrorResultEntry extends Schema.TaggedClass(...)('ToolError', {
  toolCallId: ToolCallId,                              // ← NEW
  tagName:    Schema.String,
  status:     Schema.Literals(['error','rejected','interrupted']),
  message:    Schema.NullOr(Schema.String),
}) {}

// Pre-identification parse failures — no toolCallId (we don't know which call yet)
class ToolParseErrorResultEntry      extends Schema.TaggedClass(...)('ToolParseError',      { /* existing */ }) {}
class StructuralParseErrorResultEntry extends Schema.TaggedClass(...)('StructuralParseError',{ /* existing */ }) {}

// Other entries unchanged
class MessageAckResultEntry           extends Schema.TaggedClass(...)('MessageAck',           { /* existing */ }) {}
class NoToolsOrMessagesResultEntry    extends Schema.TaggedClass(...)('NoToolsOrMessages',    { /* existing */ }) {}
// ... etc
```

### 8.2 TimelineEntry (no changes)

Timeline entries remain as they are: `user_message`, `parent_message`, `user_bash_command`, `agent_block` (subagent activity, uses existing `AgentAtom`), `presence`, `observation`, lifecycle hooks, task updates.

---

## 9. TurnPartEvent (Streaming Decode Events)

What `Codec.decode` emits while the model is generating. Granular: `start` / `delta` / `end` for each part type so memory and UI can update incrementally.

```ts
// packages/codecs/src/events/turn-part-event.ts
export const TurnPartEvent = Schema.Union([
  // Thought
  ThoughtStart,
  ThoughtDelta,
  ThoughtEnd,
  // Tool call
  ToolCallStart,
  ToolCallInputDelta,
  ToolCallEnd,
  // Bookkeeping
  TurnUsage,
  TurnFinish,
])
export type TurnPartEvent = typeof TurnPartEvent.Type
```

Sketches:

```ts
class ThoughtStart      extends Schema.TaggedClass(...)('ThoughtStart',      { id: ThoughtId, level: Schema.Literals(['low','medium','high']) }) {}
class ThoughtDelta      extends Schema.TaggedClass(...)('ThoughtDelta',      { id: ThoughtId, text: Schema.String }) {}
class ThoughtEnd        extends Schema.TaggedClass(...)('ThoughtEnd',        { id: ThoughtId }) {}

class ToolCallStart     extends Schema.TaggedClass(...)('ToolCallStart',     { id: ToolCallId, toolName: Schema.String }) {}
class ToolCallInputDelta extends Schema.TaggedClass(...)('ToolCallInputDelta',{ id: ToolCallId, jsonChunk: Schema.String }) {}
class ToolCallEnd       extends Schema.TaggedClass(...)('ToolCallEnd',       { id: ToolCallId, input: Schema.Unknown }) {}

class TurnUsage         extends Schema.TaggedClass(...)('TurnUsage',         { promptTokens: Schema.Int, completionTokens: Schema.Int, ... }) {}
class TurnFinish        extends Schema.TaggedClass(...)('TurnFinish',        { reason: Schema.Literals(['stop','tool_calls','length','content_filter','other']) }) {}
```

The memory projection consumes this stream and:
- Buffers `Thought*` events into a `ThoughtPart` until `ThoughtEnd`, then appends to the current `AssistantTurnMessage.parts`.
- Same for `ToolCall*` events into a `ToolCallPart`.
- Appends `TurnUsage`/`TurnFinish` info to the assistant turn's metadata (existing strategy / usage tracking).

The UI can subscribe to the same stream and render in real time without waiting for completion.

---

## 10. Wire Types

Per-driver typed shapes. Schema-validated where it matters (response chunks); the request side may be plain interfaces if construction is tightly controlled.

### 10.1 Chat Completions (native, xml-act)

```ts
// packages/drivers/src/wire/chat-completions.ts

export interface ChatCompletionsRequest {
  readonly model:       string
  readonly messages:    readonly ChatMessage[]
  readonly tools?:      readonly ChatTool[]
  readonly stream:      true
  readonly temperature?: number
  readonly max_tokens?:  number
  // ... etc
}

type ChatMessage =
  | { role: 'system';    content: string | readonly ContentPart[] }
  | { role: 'user';      content: string | readonly ContentPart[] }
  | { role: 'assistant'; content?: string | null; tool_calls?: readonly ChatToolCall[]; reasoning_content?: string }
  | { role: 'tool';      tool_call_id: string; content: string | readonly ContentPart[] }

interface ChatTool {
  readonly type: 'function'
  readonly function: { readonly name: string; readonly description: string; readonly parameters: unknown }
}

interface ChatToolCall {
  readonly id: string
  readonly type: 'function'
  readonly function: { readonly name: string; readonly arguments: string }  // JSON string
}

// Response chunk (SSE)
export class ChatCompletionsStreamChunk extends Schema.Class<ChatCompletionsStreamChunk>('ChatCompletionsStreamChunk')({
  id:      Schema.String,
  choices: Schema.Array(Schema.Struct({
    index:        Schema.Int,
    delta:        Schema.Struct({
      role:              Schema.optional(Schema.String),
      content:           Schema.optional(Schema.NullOr(Schema.String)),
      reasoning_content: Schema.optional(Schema.NullOr(Schema.String)),
      tool_calls:        Schema.optional(Schema.Array(/* ToolCallChunk */)),
    }),
    finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
  })),
  usage: Schema.optional(/* Usage */),
}) {}
```

### 10.2 Completions (raw text)

```ts
// packages/drivers/src/wire/completions.ts

export interface CompletionsRequest {
  readonly model:        string
  readonly prompt:       string
  readonly stream:       true
  readonly stop?:        readonly string[]
  readonly temperature?: number
  readonly max_tokens?:  number
  readonly extra_body?:  Record<string, unknown>  // for vision: { images: [...] }
}

export class CompletionsStreamChunk extends Schema.Class<CompletionsStreamChunk>('CompletionsStreamChunk')({
  choices: Schema.Array(Schema.Struct({
    text:          Schema.String,
    finish_reason: Schema.optional(Schema.NullOr(Schema.String)),
  })),
  usage: Schema.optional(/* Usage */),
}) {}
```

---

## 11. Codec Interface

A `Codec` is parameterized by the wire types it produces and consumes:

```ts
// packages/codecs/src/codec.ts

export interface Codec<WireRequest, WireChunk> {
  /**
   * Encode the current memory + tool defs into a wire request body.
   * Pure where possible — may yield an Effect for image fetching, schema canonicalization, etc.
   */
  readonly encode: (
    memory: readonly Message[],
    tools:  readonly ToolDef[],
    options: EncodeOptions,
  ) => Effect.Effect<WireRequest, CodecEncodeError>

  /**
   * Decode a stream of wire chunks into a stream of TurnPartEvents.
   * Stateful per-call; the codec maintains assembly state across chunks
   * but not across turns.
   */
  readonly decode: (
    chunks: Stream.Stream<WireChunk, DriverError>,
  ) => Stream.Stream<TurnPartEvent, CodecDecodeError | DriverError>
}

export interface EncodeOptions {
  readonly thinkingLevel?: 'low' | 'medium' | 'high'
  readonly maxTokens?:     number
  readonly stopSequences?: readonly string[]
  // ... per-call knobs
}
```

Concrete impls:

```ts
export const NativeChatCompletionsCodec: Codec<ChatCompletionsRequest, ChatCompletionsStreamChunk> = ...
export const XmlActCodec:                Codec<ChatCompletionsRequest, ChatCompletionsStreamChunk> = ...     // later
export const makeCompletionsCodec:       (adapter: ModelAdapter) => Codec<CompletionsRequest, CompletionsStreamChunk> = ...
```

`Codec` is **not** a `Context.Tag`. It's a plain interface; instances are constructed values held by `Model` records.

---

## 12. Driver Interface

Pure transport. Takes a typed request, returns a typed stream.

```ts
// packages/drivers/src/driver.ts

export interface Driver<WireRequest, WireChunk> {
  readonly id: string                          // for tracing / logging
  readonly send: (
    request: WireRequest,
    options: DriverCallOptions,
  ) => Effect.Effect<
    Stream.Stream<WireChunk, DriverError>,
    DriverError,
    HttpClient.HttpClient | TraceEmitter | Auth     // Effect-injected dependencies
  >
}

export interface DriverCallOptions {
  readonly endpoint:  string                   // e.g. "https://api.fireworks.ai/inference/v1/chat/completions"
  readonly authToken: string                   // bearer
  readonly signal?:   AbortSignal
  readonly timeoutMs?: number
}
```

Concrete impls:

```ts
export const OpenAIChatCompletionsDriver: Driver<ChatCompletionsRequest, ChatCompletionsStreamChunk> = ...
export const OpenAICompletionsDriver:     Driver<CompletionsRequest,     CompletionsStreamChunk>     = ...
```

The driver knows nothing about codecs, paradigms, memory, or tools. It just does HTTP + SSE parsing into the typed chunk stream.

---

## 13. ModelAdapter Interface

Used **only** by `CompletionsCodec`. Encodes/decodes one model family's native trained format (e.g., Kimi's `<|im_start|>functions.foo:0` style, GLM's variant, Qwen's variant, etc.).

```ts
// packages/codecs/src/adapters/model-adapter.ts

export interface ModelAdapter {
  readonly id: string                          // 'kimi' | 'glm' | 'qwen' | ...

  /**
   * Render the system-prompt fragment that declares tools to the model
   * in its native trained format.
   */
  readonly encodeTools: (
    tools: readonly ToolDef[],
  ) => string

  /**
   * Render the conversation prompt for the completions endpoint.
   * Memory + tool defs in, single string out.
   */
  readonly encodePrompt: (
    memory: readonly Message[],
    tools:  readonly ToolDef[],
    options: EncodeOptions,
  ) => string

  /**
   * Parse the streaming text output of the model into TurnPartEvents.
   * The adapter knows the family-specific token sequences for thoughts,
   * tool calls, etc.
   */
  readonly decode: (
    text: Stream.Stream<string, DriverError>,
  ) => Stream.Stream<TurnPartEvent, CodecDecodeError | DriverError>
}
```

Concrete impls live in `packages/codecs/src/adapters/{kimi,glm,qwen,minimax,deepseek}.ts`.

`CompletionsCodec` is constructed with one adapter and delegates encode/decode to it:

```ts
export const makeCompletionsCodec = (adapter: ModelAdapter): Codec<CompletionsRequest, CompletionsStreamChunk> => ({
  encode: (memory, tools, options) => Effect.sync(() => ({
    model: '...',
    prompt: adapter.encodePrompt(memory, tools, options),
    stream: true,
    stop: [/* family-specific stop sequences */],
  })),
  decode: chunks => adapter.decode(textStreamFromChunks(chunks)),
})
```

---

## 14. Model Record (Composition Node)

A `Model` ties everything together. Holds id + capabilities (data) and a Driver + Codec (behavior). No string-literal dispatch.

```ts
// packages/providers/src/model/model.ts (refactored)

export class ModelCapabilities extends Schema.Class<ModelCapabilities>('ModelCapabilities')({
  vision:    Schema.Boolean,
  reasoning: Schema.Boolean,
  tools:     Schema.Boolean,
  // ... etc
}) {}

export interface Model<WireRequest = unknown, WireChunk = unknown> {
  readonly id:           ModelId
  readonly displayName:  string
  readonly capabilities: ModelCapabilities

  readonly driver:       Driver<WireRequest, WireChunk>
  readonly codec:        Codec<WireRequest, WireChunk>

  // Per-model wire config (which endpoint, model-name-on-wire, default params)
  readonly wireConfig: {
    readonly endpoint:        string
    readonly wireModelName:   string         // value put in `model` field on wire
    readonly defaultMaxTokens: number
    // ... etc
  }
}
```

Construction example:

```ts
// Kimi K2.6 on Fireworks via native paradigm
const kimiK26Native: Model<ChatCompletionsRequest, ChatCompletionsStreamChunk> = {
  id: ModelId.makeUnsafe('kimi-k2.6:fireworks:native'),
  displayName: 'Kimi K2.6 (Fireworks, native)',
  capabilities: new ModelCapabilities({ vision: true, reasoning: true, tools: true }),
  driver: OpenAIChatCompletionsDriver,
  codec:  NativeChatCompletionsCodec,
  wireConfig: {
    endpoint: 'https://api.fireworks.ai/inference/v1/chat/completions',
    wireModelName: 'accounts/fireworks/models/kimi-k2p6',
    defaultMaxTokens: 4096,
  },
}

// Same model, completions paradigm
const kimiK26Completions: Model<CompletionsRequest, CompletionsStreamChunk> = {
  id: ModelId.makeUnsafe('kimi-k2.6:fireworks:completions'),
  displayName: 'Kimi K2.6 (Fireworks, completions)',
  capabilities: new ModelCapabilities({ vision: true, reasoning: true, tools: true }),
  driver: OpenAICompletionsDriver,
  codec:  makeCompletionsCodec(KimiAdapter),
  wireConfig: {
    endpoint: 'https://api.fireworks.ai/inference/v1/completions',
    wireModelName: 'accounts/fireworks/models/kimi-k2p6',
    defaultMaxTokens: 4096,
  },
}
```

The catalog (existing `packages/providers/src/catalog/`) holds **factory functions** for these `Model` records, parameterized by auth/connection info. Resolution happens once at session start; the resolved `Model` is then carried through.

---

## 15. TurnEngine (Orchestrator)

Effect service (`Context.Tag`) that runs the turn loop.

```ts
// packages/agent/src/engine/turn-engine.ts

export interface TurnEngineShape {
  readonly runTurn: (params: {
    readonly model:    Model
    readonly memory:   readonly Message[]
    readonly tools:    readonly ToolDef[]
    readonly options:  EncodeOptions
    readonly signal?:  AbortSignal
  }) => Effect.Effect<
    Stream.Stream<TurnPartEvent, RunTurnError>,
    RunTurnError,
    HttpClient.HttpClient | TraceEmitter | ToolExecutor | Memory
  >
}

export class TurnEngine extends Context.Tag('TurnEngine')<TurnEngine, TurnEngineShape>() {}
```

Conceptually, `runTurn`:

1. Calls `model.codec.encode(memory, tools, options)` → `WireRequest`.
2. Calls `model.driver.send(request, ...)` → `Stream<WireChunk>`.
3. Calls `model.codec.decode(chunks)` → `Stream<TurnPartEvent>`.
4. Returns the event stream to the caller (memory/UI both subscribe).

A higher-level function `runUntilStop` may chain turns: after each turn, if any `ToolCallEnd` events occurred, executes those tools (via `ToolExecutor`), appends results to memory's inbox, and runs another turn — until `TurnFinish.reason !== 'tool_calls'`.

```ts
// pseudo
const runUntilStop = (model, memory, tools, options) => Effect.gen(function*() {
  let currentMemory = memory
  while (true) {
    const events = yield* turnEngine.runTurn({ model, memory: currentMemory, tools, options })
    const turnOutcome = yield* collectTurn(events)   // updates memory with new AssistantTurnMessage
    currentMemory = turnOutcome.newMemory
    if (turnOutcome.toolCalls.length === 0) break
    const results = yield* executeTools(turnOutcome.toolCalls)
    currentMemory = appendInboxWithResults(currentMemory, results)
  }
  return currentMemory
})
```

The key point is the engine doesn't know which paradigm it's running. All paradigm-specific behavior is encapsulated in the codec.

---

## 16. Effect Services

All `Context.Tag` services involved:

| Service | Defined in | Used by | Purpose |
|---|---|---|---|
| `HttpClient.HttpClient` | `effect/unstable/http` | drivers | HTTP transport. |
| `TraceEmitter` | existing `providers/resolver/tracing.ts` | drivers, codecs | Driver-facing tracing. |
| `TracePersister` | existing `providers/resolver/tracing.ts` | agent / TurnEngine | Agent-facing tracing. |
| `Auth` | existing `providers/auth/` | drivers | Resolves API keys / OAuth tokens. |
| `ToolExecutor` | new in agent | TurnEngine | Executes tool calls and produces ResultEntries. |
| `Memory` | new (agent projection wrapper) | TurnEngine | Read/write Message[]. May simply be a function passed in rather than a tag — TBD. |
| `TurnEngine` | new | agent top-level | Drives turns. |

Codecs themselves are **not** `Context.Tag`s. They're values held by `Model` records and called directly via `model.codec.encode(...)` / `model.codec.decode(...)`.

---

## 17. Lifecycle: Encoding a Turn

`codec.encode(memory, tools, options) → WireRequest`. Below is the **native** codec on chat completions; other codecs follow analogous patterns.

```
For each Message in memory (in order):
  case SessionContextMessage / ForkContextMessage / CompactedMessage:
    → emit { role: 'system', content: textOf(content) }   // or 'developer' for o-series

  case AssistantTurnMessage:
    → from parts, derive a single chat-message:
        - thoughts → reasoning_content (joined)
        - tool calls → tool_calls[] (id, name, arguments-as-JSON-string)
        - content: null
      emit { role: 'assistant', content: null, reasoning_content: ..., tool_calls: [...] }

  case InboxMessage:
    For each result entry:
      case ToolObservation { toolCallId, content }:
        → emit { role: 'tool', tool_call_id: toolCallId, content: content }   // ContentPart[] passed through
      case ToolError { toolCallId, message }:
        → emit { role: 'tool', tool_call_id: toolCallId, content: messageOrPlaceholder }
      case ToolParseError / StructuralParseError / MessageAck / ...:
        → either accumulate as a system or user message, or render textually inline (TBD per entry)

    For timeline entries:
      → render to a single user-role message (multimodal if any image attachments)
        with sections per entry (similar to existing inbox rendering)

Append `tools` array if any toolDefs exist:
  tools: toolDefs.map(td => ({ type: 'function', function: { name: td.name, description: td.description, parameters: td.parameters } }))

Set top-level: { model: wireModelName, messages: [...], tools: [...], stream: true, temperature, max_tokens, ... }
```

Notes:

- `reasoning_content` may not be supported by all servers. The codec inspects model capabilities and either includes thoughts as `reasoning_content` (if supported) or drops them from the encoded request (since they're already "in" the model's context via prior turn tokens conceptually — or, for some providers, included as text inside `content`). TBD: choose a policy. Open question §28.
- ContentPart arrays in tool messages: passed through as-is (we tested this works on Fireworks). For providers that reject multimodal tool messages, the codec falls back to extracting images into a follow-up user message and putting only text in the tool message. This fallback is configurable per-`Driver` or per-`Model` (§28).

---

## 18. Lifecycle: Decoding a Turn

`codec.decode(chunks) → Stream<TurnPartEvent>`. The decoder is stateful within a turn.

For the **native chat completions** codec, the decoder maintains:

- A single open `ThoughtPart` (when a chunk contains `reasoning_content`).
- Zero or more in-progress `ToolCallPart`s (keyed by index in the streamed `tool_calls[]` arrays — model emits tool calls progressively).

Per chunk:

```
delta.reasoning_content present:
  if no thought open → emit ThoughtStart, then ThoughtDelta(text)
  else                → emit ThoughtDelta(text)

delta.content present (rare in pure tool-calling flows):
  → either route into an ongoing thought, or treat as end-of-thought boundary
  (decision: §28 — for now, treat content as ending any open thought, then ignore content text since "message" tag is gone)

delta.tool_calls[i] present:
  if no tool call open at index i → emit ToolCallStart(toolName from delta.function.name), record id (we generate)
  delta.function.arguments → emit ToolCallInputDelta(jsonChunk)

finish_reason present:
  → emit ThoughtEnd (if any thought is open)
  → for each open tool call: parse accumulated arguments → emit ToolCallEnd(parsed input)
  → emit TurnUsage (if usage chunk present)
  → emit TurnFinish(reason)
```

The decoder ignores server-provided `tool_calls[i].id` and uses a locally generated `ToolCallId` instead.

---

## 19. Lifecycle: Tool Execution

After a turn ends with `finish_reason: 'tool_calls'`, `TurnEngine` (or its wrapper) executes each tool call.

```ts
interface ToolExecutorShape {
  readonly execute: (
    call: { toolCallId: ToolCallId; toolName: string; input: unknown },
  ) => Effect.Effect<ResultEntry, ToolExecutionError>
}
export class ToolExecutor extends Context.Tag('ToolExecutor')<ToolExecutor, ToolExecutorShape>() {}
```

Results (`ResultEntry`s) are appended to a new `InboxMessage` after the assistant turn. Then the loop continues with another `runTurn`.

Tool execution is parallelizable across calls. Order preservation in the inbox is by tool-call ordinal within the assistant turn.

---

## 20. Paradigm: Native (Chat Completions)

End-to-end path for one turn:

1. **Encode**: `NativeChatCompletionsCodec.encode(memory, tools, options)` → `ChatCompletionsRequest`
   - System messages from session/fork context
   - Assistant turn messages → `{ role: 'assistant', tool_calls, reasoning_content }`
   - Inbox results → tool messages (with our `toolCallId`) and user messages
   - Tools → `tools[]` array
2. **Send**: `OpenAIChatCompletionsDriver.send(request, { endpoint, authToken })` → SSE stream → `Stream<ChatCompletionsStreamChunk>`
3. **Decode**: `NativeChatCompletionsCodec.decode(chunks)` → `Stream<TurnPartEvent>`
4. **Apply**: Memory projection consumes the event stream, builds `TurnPart[]`, appends `AssistantTurnMessage`. UI subscribes to the same stream.
5. **Tool exec**: If tool calls, `ToolExecutor` runs them, results appended as `InboxMessage`. Loop.

This is the **first** paradigm we'll implement. Verifies the whole architecture end-to-end.

---

## 21. Paradigm: Completions (Raw Text + Adapter)

Same architecture, different codec/driver pair:

1. **Encode**: `CompletionsCodec(KimiAdapter).encode(...)`. Adapter's `encodePrompt` produces a single string in the family's chat template (e.g., `<|im_start|>system\n... \n<|im_end|>...`). Tools rendered into the system prompt by `adapter.encodeTools`.
2. **Send**: `OpenAICompletionsDriver.send({ prompt, ... })` → text-token stream.
3. **Decode**: `CompletionsCodec.decode` → `adapter.decode(stream)` parses the family's tokens (e.g., when it sees `<|im_start|>functions.foo:0`, that's a tool call start).
4. **Apply / Tool exec**: same as native.

The whole "this model family emits this token to start a tool call" complexity is contained inside the `ModelAdapter`. Codec, driver, engine, memory, UI: unchanged.

Implementation order: implement adapters one family at a time (Kimi → GLM → Qwen → MiniMax → DeepSeek). Each family has a `MODEL.md` already documenting its native format (existing).

---

## 22. Paradigm: xml-act (Universal Text Tags)

**Last** to port. Behaves like a codec on chat completions:

1. **Encode**: `XmlActCodec.encode` renders all of memory as text:
   - `messages[].content` is plain text containing magnitude tags
   - System message contains tool schema as XML
   - Assistant messages contain rendered XML (thoughts, tool calls, etc.)
   - User messages contain rendered inbox XML
2. **Send**: `OpenAIChatCompletionsDriver` (same driver as native).
3. **Decode**: `XmlActCodec.decode` runs the existing tag parser over the assistant content text stream, emits `TurnPartEvent`s.
4. **Apply / Tool exec**: same.

Open question (§28): once removing `<message>` tag and yield syntax, does xml-act simplify enough that this is straightforward? Or do we want to delete xml-act entirely once native + completions cover everything?

---

## 23. ID Generation & Pairing

We generate `ToolCallId`s. Format reuses existing parser: `call-{ord}-{ts36}`. Generated by:

- The codec's decode loop on `ToolCallStart` (we ignore server IDs).
- Or, equivalently, by the encode-side when constructing tool messages — if we have to round-trip a server-provided ID it'd be wrong; we always use ours.

Pairing rules:

- An `AssistantTurnMessage.parts[i] = ToolCallPart { id: X }` is paired with an `InboxMessage.results[j] = ToolObservationResultEntry { toolCallId: X }` by exact ID match.
- The codec's encode looks up pairings to emit `tool_call_id: X` on the wire tool message.
- Out-of-order results work: codec re-orders if necessary on encode (most providers don't care about strict ordering as long as IDs match; some do require tool result immediately after corresponding tool call — Anthropic does, OpenAI is lax).

Verified with Fireworks/Kimi K2.6: passed `tc_001_made_up_by_us` as our ID, server accepted and continued correctly.

---

## 24. Multimodal Handling per Paradigm

| Paradigm | User msg images | Assistant msg images | Tool msg images |
|---|---|---|---|
| **xml-act** | ContentPart[] in text+image multimodal user message | N/A (assistant is text-only) | embedded in user-role inbox message |
| **native** | ContentPart[] in `content` array | N/A (no images in assistant) | **passes through in tool message content** (verified Fireworks/Kimi); fallback: text-only tool message + follow-up user message with image |
| **completions** | embedded via `extra_body.images: [...]` array, with `<image>` token in prompt | N/A | adapter renders into prompt tokens with `<image>` markers |

Codec implementations expose a config hook for the multimodal-tool fallback so we can flip between "send images in tool message" vs "extract to follow-up user message" per provider if needed.

---

## 25. Error Handling

All errors are `Schema.TaggedError` classes:

```ts
class CodecEncodeError extends Schema.TaggedErrorClass(...)('CodecEncodeError', { reason: Schema.String, context: Schema.Unknown }) {}
class CodecDecodeError extends Schema.TaggedErrorClass(...)('CodecDecodeError', { reason: Schema.String, partial: Schema.Unknown }) {}
class DriverError      extends Schema.TaggedErrorClass(...)('DriverError',      { reason: Schema.String, status: Schema.optional(Schema.Int), body: Schema.Unknown }) {}
class ToolExecutionError extends Schema.TaggedErrorClass(...)('ToolExecutionError', { toolName: Schema.String, message: Schema.String }) {}
class RunTurnError = CodecEncodeError | DriverError | CodecDecodeError | ToolExecutionError
```

Decode errors are non-fatal where possible: a malformed chunk emits a `CodecDecodeError` event into the stream, but the stream may continue if subsequent chunks recover. Unrecoverable errors abort the stream.

---

## 26. Tracing

Drivers emit `TraceInput` to the `TraceEmitter` for each request/response. Codecs may also emit traces (e.g., `codec.encode` produces a trace with the wire request body for debugging). The existing `withTraceScope` enrichment to `TraceData` flows to the persister.

This is unchanged from existing patterns in `packages/providers/src/resolver/tracing.ts`.

---

## 27. Implementation Plan

Phased to deliver native end-to-end first, then completions, then port xml-act.

### Phase 0: Contracts (this spec → code)

- Create `packages/codecs/` skeleton.
- Create `packages/drivers/` skeleton.
- Define all branded IDs.
- Define `Message`, `TurnPart`, `ResultEntry`, `TimelineEntry` schemas.
- Define `TurnPartEvent` schema.
- Define `ToolDef` schema.
- Define `Codec` and `Driver` interfaces.
- Define `ModelAdapter` interface.
- Define wire types for chat completions (request + chunk).
- Define `Model` shape.

No implementations yet — just types and interfaces. Let TypeScript compilation pass on the skeleton.

### Phase 1: Native end-to-end

- Implement `OpenAIChatCompletionsDriver` (SSE parsing, HttpClient integration, auth, tracing).
- Implement `NativeChatCompletionsCodec` (encode + decode).
- Build a Kimi K2.6 / Fireworks `Model` record using the above.
- Implement `TurnEngine.runTurn`.
- Wire memory projection to consume `TurnPartEvent`s and produce `AssistantTurnMessage`s.
- Wire inbox projection to consume tool execution results with `toolCallId`.
- Run end-to-end against Fireworks. Manual + automated tests.

Success criteria: an agent can hold a conversation, call tools, and respond, using only the native paradigm against Fireworks/Kimi.

### Phase 2: Completions paradigm

- Implement `OpenAICompletionsDriver`.
- Implement `CompletionsCodec(adapter)` shell.
- Implement `KimiAdapter` (highest priority — same model we tested in Phase 1 but raw completions).
- Build a Kimi-completions `Model` record.
- Test against same Fireworks endpoint, completions API.

Success criteria: same agent, same tests, but using completions paradigm. Validate Phase 1 abstractions hold up.

### Phase 3: Additional adapters

- `GLMAdapter`, `QwenAdapter`, `MiniMaxAdapter`, `DeepSeekAdapter` — one at a time.
- Each adds models without changing any other code.

### Phase 4: xml-act port

- Implement `XmlActCodec` over the existing parser/renderer.
- Reuse `OpenAIChatCompletionsDriver`.
- Validate against existing test suite.
- Decommission old non-codec wiring.

### Phase 5: Surface cleanup

- Remove filter system, named lenses, yield syntax, `<message>` tag — items the user has already greenlit dropping.
- `send_message_to_user` becomes a tool. `send_message_to_worker` becomes a tool.
- Update prompts, tool schemas, and UI to match.

---

## 28. Open Questions

1. **`reasoning_content` policy.** Always include in encode? Or skip if model's prior turn already has it baked in via the model's own context? Probably: include only the most recent turn's thoughts as a hint, drop older. Decision needed.
2. **Multimodal-tool-message fallback**: keep code path even though Fireworks works? Probably yes, for non-Fireworks providers.
3. **`query` field on `ToolCallPart`**: the previous "filter" system is dropped. Some tools used a query/filter at call time (e.g., grep). Does that survive as a tool argument (cleaner) or stay as a separate `query` field? Lean toward: drop the field, fold into tool input.
4. **Drop `<message>` from xml-act before porting?** Or port the existing xml-act first and clean up later? Lean: clean up first; less to port.
5. **`role: 'developer'` vs `role: 'system'`** for context messages. Newer OpenAI models prefer `developer`. Codec should pick based on `Model.capabilities` or wire config.
6. **Whether `TurnEngine` should be the loop driver, or just a single-turn primitive** with a separate `runUntilStop` higher-level helper. Current draft: `runTurn` only; `runUntilStop` is a utility on top. Confirm.
7. **Memory as Effect service vs plain function arg.** TurnEngine could either pull `Memory` from context (`Context.Tag`) or take it as a parameter. The latter is simpler; the former is testable. Lean: parameter.
8. **Tool result content stored as `ContentPart[]` vs `string`.** Current decision: `ContentPart[]` (multimodal-aware). Confirmed empirically on Fireworks. No regression risk.
9. **Where do model "templates" (existing `packages/providers/src/model/generated/templates/`) live?** Probably move under `packages/codecs/src/adapters/<family>/` since the templates are family-specific. To revisit during Phase 2.

---

## 29. Appendix: Mapping to Existing Code

| New | Existing equivalent / source | Notes |
|---|---|---|
| `Message` (codecs/memory) | `packages/agent/src/projections/memory.ts` | Refactor `assistant_turn` to hold `parts: TurnPart[]` instead of `content: ContentPart[]`. Inbox unchanged. |
| `ContentPart` | `@magnitudedev/tools` | Re-export. |
| `TurnPart` | new | Replaces ad-hoc text in `assistant_turn.content`. |
| `TurnPartEvent` | partially `packages/agent/src/inbox/types.ts` (`AgentAtom`) | `AgentAtom` is for **subagent** activity in parent view — different concept. `TurnPartEvent` is the codec's streaming output. Don't conflate. |
| `ResultEntry` | `packages/agent/src/inbox/types.ts` | Add `toolCallId` to `ToolObservationResultItem`, `ToolErrorResultItem`. |
| `TimelineEntry` | `packages/agent/src/inbox/types.ts` | No change. |
| `ToolDef` | partially `packages/tools/src/state-model.ts` | Schema-ify for codec consumption. |
| `Codec` | new | No existing equivalent. |
| `Driver` | `packages/providers/src/drivers/types.ts` (`ExecutableDriver`) | Existing driver interface is BAML-coupled and returns `Stream<string>`. New `Driver` interface returns typed wire chunks. New, not refactor. |
| `ModelAdapter` | partially `packages/model-adapters/families/{kimi,glm,qwen,minimax,deepseek}/MODEL.md` | Docs become runtime adapters. |
| `Model` | `packages/providers/src/model/model.ts` (`ProviderModel`, `BoundModel`) | Refactor to hold codec + driver instances directly. |
| `TurnEngine` | partially `packages/agent/src/execution/execution-manager.ts` | Will reshape into a service. |
| `ToolExecutor` | partially `packages/agent/src/tools/` | Refactor to produce `ResultEntry[]`. |
| Branded IDs | partially `packages/xml-act/src/parser/...generateId()` | Same format `call-{ord}-{ts36}`. Codify via `Schema.brand`. |
| `TraceEmitter`, `TracePersister` | existing `packages/providers/src/resolver/tracing.ts` | Reuse. |
| Wire types | new | No existing equivalent (BAML driver hides them). |

---

## End of spec

Sign-off needed on:
- Storage shape (§6, §7, §8)
- Codec/Driver/ModelAdapter interfaces (§11, §12, §13)
- Model record composition (§14)
- Implementation phasing (§27)
- Open questions (§28)

Once those are settled, Phase 0 begins.
