# Native Paradigm — Implementation Plan

**Status:** Draft for review
**Date:** 2026-04-27
**Spec:** [`codec-driver-contracts.md`](./codec-driver-contracts.md)
**Research reports** (in workspace, not committed):
- `$M/reports/native-research-memory-turn.md`
- `$M/reports/native-research-execution-path.md`
- `$M/reports/native-research-tools-inbox.md`
- `$M/reports/native-research-providers.md`
- `$M/reports/native-research-display-cli.md`

---

## Table of Contents

1. [Goal & Success Criteria](#1-goal--success-criteria)
2. [Scope](#2-scope)
3. [Architecture](#3-architecture)
4. [Strategic Decisions](#4-strategic-decisions)
5. [Phase 0 — Scaffold packages & contracts](#5-phase-0--scaffold-packages--contracts)
6. [Phase 1 — `OpenAIChatCompletionsDriver`](#6-phase-1--openaichatcompletionsdriver)
7. [Phase 2 — `NativeChatCompletionsCodec`](#7-phase-2--nativechatcompletionscodec)
8. [Phase 3 — Canonical event vocabulary refactor](#8-phase-3--canonical-event-vocabulary-refactor)
9. [Phase 4 — Memory & projection refactor](#9-phase-4--memory--projection-refactor)
10. [Phase 5 — TurnEngine + ToolDispatcher + Cortex rewrite](#10-phase-5--turnengine--tooldispatcher--cortex-rewrite)
11. [Phase 6 — Provider wiring & end-to-end](#11-phase-6--provider-wiring--end-to-end)
12. [Risks, verification, decisions log](#12-risks-verification-decisions-log)

> **Iteration note (2026-04-27, post-draft).** This plan has been corrected in place after multiple codebase verification passes. Key corrections:
> - Events in `packages/agent/src/events.ts` are **plain TypeScript interfaces** with a `type: string` discriminator and `forkId: string | null` — NOT `Schema.TaggedClass`. The plan now matches.
> - Errors use `Schema.TaggedError<X>()(...)` (parameterless first-call) per `packages/agent/src/services/web-search-service.ts`. NOT `Schema.TaggedErrorClass`. Confirmed.
> - Codec-internal types use `Schema.TaggedClass<X>('X')('tag', { ... })` per `node_modules/effect/dist/dts/Schema.d.ts` — first parens are the optional identifier, second is `(tag, fields)`.
> - `Schema.Literal('a', 'b', 'c')` is the variadic form. NOT `Schema.Literals(...)`. Confirmed.
> - Workers in `@magnitudedev/event-core` receive `publish: PublishFn<TEvent>` and `read: WorkerReadFn<TEvent>` as **parameters** to event handlers — NOT via `Context.Tag`. `TurnEngine.runTurn` now returns `Stream<TurnPartEvent>`; Cortex's worker handler drains the stream and publishes via the injected `publish` function. `ToolDispatcher.dispatch{,All}` similarly accept `publish` as a parameter.
> - The existing `TurnOutcome` union in `packages/agent/src/events.ts` is rich (`Completed | ParseFailure | ProviderNotReady | ConnectionFailure | ContextWindowExceeded | OutputTruncated | SafetyStop | Cancelled | UnexpectedError`). We **keep this union as-is** and map codec/driver errors into the appropriate variants. Earlier draft incorrectly proposed simplifying it. The plan now retains fidelity; only the `Completed` variant gains `toolCallsCount + finishReason` (replacing `yieldTarget`).
> - Assistant *content text* (the model's user-facing prose response, not thoughts) IS persisted as a `MessagePart` in `TurnPart`. The earlier draft's D15 (treat content text as ephemeral) was wrong — when an agent emits a final text response with no tool calls, that text is part of the conversation and must round-trip on subsequent turns. `TurnPart` is now `ThoughtPart | MessagePart | ToolCallPart`. The encoder maps `MessagePart` to `{ role: 'assistant', content: string }` (combined with thoughts and tool calls in the same wire message).
> - Implicit turn control: the **model never emits a yield**. Halting is `outcome._tag === 'Completed' && toolCallsCount === 0`. Worker-→-parent / agent-→-user routing uses dedicated `send_message_to_*` tools (R6 in §12.1).
> - `Driver` consumes `HttpClient.HttpClient` (from `@effect/platform`) as a Tag; the runtime root provides it via `FetchHttpClient.layer` (from `@effect/platform`, already used in `packages/agent/src/coding-agent.ts:57`). Bun ships native `fetch`, so `FetchHttpClient.layer` works on Bun without a Bun-specific package. No bespoke `fetch` wrangling; no `signal`/`timeoutMs` on `DriverCallOptions` — Effect interruption is the cancellation path. (Earlier draft incorrectly proposed `BunHttpClient.layer` from `@effect/platform-bun` — that export does not exist.)
> - `HttpClientResponse.stream(effect)` is a **module-level helper**, not a property on the response object. The driver pseudocode now reflects this: status check happens inside the response Effect, then the helper lifts to a `Stream<Uint8Array>`.
> - `Codec.decode` uses `Stream.mapAccum` with **pure state** (immutable rebuilds) — not a mutable closure. Effect streams may be consumed concurrently or replayed; mutable state is unsafe.
> - Encoder for `AssistantTurnMessage` produces `content` (from MessagePart concatenation), `reasoning_content` (from ThoughtParts), and `tool_calls` (from ToolCallParts) — all conditional on the relevant parts being present.
> - Smoke test uses the real Live stack (with `FetchHttpClient.layer`), NOT `TestHarness` (which mocks turn engine via `MockTurnScript`).
> - `Driver.send`'s Effect requirements are reduced to **`HttpClient.HttpClient` only**. The earlier draft's `TraceEmitter` requirement is **dropped** for Phase 1 — extending `TraceEvent` requires modifying `@magnitudedev/tracing` (out of this plan's scope). Driver-side tracing is a follow-up. The driver does **not** emit any traces in this plan.
> - Effect tag style: existing `ModelResolver` is `Context.GenericTag<ModelResolverShape<string>>('ModelResolver')`. **Keep this style** for the refactor; do not migrate to `Context.Tag`. New Tags introduced in the plan (`TurnEngine`, `ToolDispatcher`, `ToolRegistry`, `ProtocolBindings`) use the modern `class X extends Context.Tag('X')<X, Shape>() {}` style; that's fine — both styles co-exist in the codebase.

---

## 1. Goal & Success Criteria

**Goal.** Get the native paradigm running end-to-end against Fireworks/Kimi K2.6, with **clean** core contracts (`Driver`, `Codec`, `Model`, `TurnEngine`, `ToolDispatcher`). No translation layers, no paradigm branches in the agent loop, no shortcut wiring. xml-act path is allowed to break and is left orphaned — we are not migrating it onto the new contracts now.

**Success criteria.**

1. New packages `packages/codecs/` and `packages/drivers/` exist with all spec-required types and one concrete impl each (`NativeChatCompletionsCodec`, `OpenAIChatCompletionsDriver`).
2. The agent's turn loop is driven by a new `TurnEngine` Effect service that takes a `BoundModel`, a `ToolSet`, and the current memory, and produces a `Stream<TurnPartEvent>`.
3. The agent's tool execution is owned by a new `ToolDispatcher` Effect service, paradigm-agnostic, consumed by `TurnEngine`.
4. `Memory.assistant_turn` carries `parts: TurnPart[]` (`ThoughtPart | ToolCallPart`). `ResultEntry` tool items carry `toolCallId`. The agent's projections consume `TurnPartEvent` directly — no translation to legacy event names.
5. The CLI renders correctly (Display projection refactored to consume the canonical event vocabulary). All streaming UI works incrementally.
6. End-to-end: a multi-turn conversation against Fireworks/Kimi K2.6 with thoughts, parallel tool calls, and tool results all working.
7. Native unit tests, integration tests, and a live e2e test pass. xml-act tests may fail; that's accepted.
8. No legacy event names (`thinking_chunk`, `message_chunk`, `tool_event`, `lens_*`, `raw_response_chunk`) are emitted on the native path. The AppEvent vocabulary is unified around TurnPart semantics.

---

## 2. Scope

### In scope

- New packages: `packages/codecs/`, `packages/drivers/` with all contract types from the spec.
- One driver: `OpenAIChatCompletionsDriver` (HTTP + SSE, Effect-native, traced).
- One codec: `NativeChatCompletionsCodec` (encode + decode).
- Refactor of `MemoryProjection` to use `parts: TurnPart[]` for `assistant_turn`.
- Refactor of `DisplayProjection` to consume the canonical `TurnPartEvent` AppEvent vocabulary.
- New `TurnEngine` Effect service (`Context.Tag`) that owns the encode → send → decode loop.
- New `ToolDispatcher` Effect service (`Context.Tag`) that runs tool calls.
- Rewrite of `Cortex.turn_started` handler — no paradigm branching; calls `TurnEngine.runTurn` and `ToolDispatcher.dispatch` directly.
- Refactor of `BoundModel` to hold `driver: Driver` + `codec: Codec` + `wireConfig` as values.
- Refactor of `ModelConnection`, `ModelDriverId`, `ModelResolver` to compose codec + driver from a Model record.
- AppEvent vocabulary refactor: `TurnPartEvent` variants become first-class AppEvents; legacy xml-act event names are removed from the AppEvent union.
- Add `toolCallId` to `ToolObservationResultItem` and `ToolErrorResultItem`.
- One Kimi K2.6 / Fireworks `Model` record wired as the agent's default model.
- Native-path agent role: minimal system prompt (role text + skills), tools declared via wire `tools: [...]`.

### Out of scope (explicit)

- **Porting xml-act onto the new contracts.** No `XmlActCodec`. xml-act packages (`packages/xml-act/`) are orphaned but not deleted. xml-act-coupled projections (`canonical-turn.ts`, `canonical-xml.ts`, `replay.ts`) are unhooked from the live event flow but not deleted. xml-act tests may fail.
- **Implementing the completions paradigm or any `ModelAdapter` impl.** Interface declared (Phase 0); no impls.
- **BAML retirement as a project.** `BamlDriver` and `client-registry-builder.ts` and `model-function.ts` (BAML function names) are unhooked from the live path. They may stop compiling if shared types they depend on are refactored. We don't fix that — it's orphan code.
- **Catalog, auth, detect, usage, tracing, errors, state, runtime, browser-models** — none of these change.
- **CLI components.** CLI components consume `DisplayState` from `DisplayProjection`. We refactor Display's *inputs* (event handlers) but its *output shape* (`DisplayState`, `DisplayMessage`, `ThinkBlockMessage`, etc.) stays identical — so CLI components don't change.
- **`<message>` tag replacement with `send_message` tool, filter system removal, lenses removal.** These are surface concerns the codec simply doesn't emit. Code that handles them in xml-act stays orphaned. We don't actively delete it.
- **Compaction worker logic refactor.** Compaction's BAML call site (`CodingAgentCompact`) becomes orphan code. Compaction will need a similar codec-driven rewrite later. For this plan, we either (a) leave compaction worker broken, or (b) stub it to no-op. **Decision: stub to no-op** — compaction is non-essential for the smoke test; full rewrite is its own effort.

---

## 3. Architecture

```
                         User input
                              ↓
                         Event bus
                              ↓
                       TurnController (unchanged — pure projection-gate logic)
                              ↓ TurnStarted
                            Cortex
                              ↓
                       TurnEngine.runTurn(boundModel, memory, tools, options)
                              ↓
              ┌───────────────┴──────────────┐
              │ codec.encode(memory, tools)  │  → WireRequest
              │ driver.send(request, opts)   │  → Stream<WireChunk>
              │ codec.decode(chunkStream)    │  → Stream<TurnPartEvent>
              └───────────────┬──────────────┘
                              ↓
                      Stream<TurnPartEvent>
                              ↓
              ┌───────────────┴──────────────────────┐
              │ For each event:                       │
              │   - Publish as AppEvent (events ARE   │
              │     TurnPartEvents — no translation)  │
              │   - On ToolCallEnd: ToolDispatcher    │
              │     dispatches the call               │
              │     → publishes ToolObservation       │
              │   - On TurnFinish: Cortex publishes   │
              │     TurnOutcome AppEvent              │
              └──────────────────────────────────────┘
                              ↓
              ┌───────────────┴────────────┐
              │ MemoryProjection           │ → folds TurnPartEvent variants into
              │                            │   assistant_turn.parts; folds
              │                            │   ToolObservation into inbox results
              │ DisplayProjection          │ → folds TurnPartEvent + ToolObservation
              │                            │   into DisplayState; CLI re-renders
              │ TurnProjection             │ → unchanged
              │ AgentRouting/Status/...    │ → unchanged
              └────────────────────────────┘
                              ↓ TurnOutcome
                       (loop continues via TurnController)
```

### Key architecture properties

- **Single canonical event vocabulary.** `TurnPartEvent` (`ThoughtStart/Delta/End`, `MessageStart/Delta/End`, `ToolCallStart/InputDelta/End`, `TurnUsage`, `TurnFinish`) plus `ToolObservation` (tool result) are the AppEvents emitted on the assistant-turn execution path. No paradigm-specific event names in the live system.
- **No paradigm branching in cortex.** Cortex calls `TurnEngine.runTurn(boundModel, ...)`. The model's bound driver+codec dictate the wire format; cortex doesn't care.
- **Tool dispatch is its own service.** `ToolDispatcher` is an Effect service that knows how to run tools. `TurnEngine` invokes it on `ToolCallEnd`. No tool-execution logic mixed into the codec or the cortex.
- **BoundModel composition.** `BoundModel = { providerModel, codec, driver, wireConfig, auth }`. Constructed once by `ModelResolver` from the catalog + provider definition. Composition over dispatch.
- **xml-act is orphaned, not deleted.** Source files stay. They're not imported by anything on the live path. They may not compile after Phase 4 — accepted.

### Why this is the clean approach (vs the prior translator design)

The prior draft proposed a `TurnPartEvent → legacy AppEvent` translator at the cortex boundary so Display could continue consuming `thinking_chunk`/`message_chunk`/`tool_event` unchanged. That was a shortcut. It would have:

- Locked the codebase to xml-act-shaped event names forever.
- Created two parallel event vocabularies to maintain.
- Added a per-turn translation step on the hot path.
- Made future paradigms (xml-act port, completions) need to also emit legacy event names.

The clean answer is: the canonical event vocabulary is `TurnPartEvent`. Display projection refactors to consume those events. CLI components don't care because Display's *output shape* is preserved.

---

## 4. Strategic Decisions

| # | Decision | Rationale |
|---|---|---|
| **D1** | Codec emits `TurnPartEvent`. `TurnPartEvent` variants are first-class AppEvents (added to the AppEvent union). No translator. | One canonical vocabulary. Future paradigms emit the same events. |
| **D2** | `MemoryProjection` and `DisplayProjection` consume `TurnPartEvent` AppEvents directly. Their handler vocabulary updates; their output shapes (`Message`, `DisplayState`) preserve external contracts. | CLI components keep working. Internal refactor stays scoped to projections. |
| **D3** | `assistant_turn` holds `parts: TurnPart[]` (`ThoughtPart \| MessagePart \| ToolCallPart`). `content: ContentPart[]` is removed. **`MessagePart` carries the assistant's user-facing text response** (when the model emits content text); persisted so the model sees its prior response on the next turn. | Spec §7 + correctness — assistant text is part of the conversation, not ephemeral. |
| **D4** | `ToolObservationResultItem` and `ToolErrorResultItem` gain `toolCallId: string`. Wire-level pairing on encode is by ID. | Required for parallel tool calls and native API correctness. |
| **D5** | `TurnEngine` is an Effect service (`Context.Tag`) owning the encode → send → decode → dispatch tools loop. Replaces the xml-act-coupled `ExecutionManager.execute(xmlStream)`. `ExecutionManager` is orphaned. | Per spec §15. Clean ownership. |
| **D6** | `ToolDispatcher` is an Effect service (`Context.Tag`) owning tool execution. Knows how to map `ToolCallEnd { toolName, input }` to a `RegisteredTool` and run it under the fork's execution layer. Emits `ToolObservation` on completion. | Tool exec is orthogonal to paradigm. Extracting it lets `TurnEngine` stay focused. Spec §19. |
| **D7** | `BoundModel` holds `codec: Codec<unknown,unknown>`, `driver: Driver<unknown,unknown>`, `wireConfig: { endpoint, wireModelName, defaultMaxTokens }`, `auth: AuthInfo`, plus the existing `model: ProviderModel` and `canonicalModel: CanonicalModel \| null` fields. The legacy `invoke(CodingAgentChat, ...)`, `stream`, `complete` methods are removed. Turn execution moves to `TurnEngine` (a `Context.Tag` service), which receives a `BoundModel` and produces `Stream<TurnPartEvent>`. `BoundModel` itself has no behavior methods — it's a pure data record. | Composition node per spec §14. No BAML detritus on the live interface. |
| **D8** | `ModelConnection` has one shape: `{ auth, baseUrl }`. The `Baml` tagged variant is removed. Drivers receive auth + baseUrl explicitly. | No tag dispatch. Clean. |
| **D9** | `ModelDriverId` is just an opaque string identifier used in catalog/registry data. Driver resolution is by identity (the catalog entry references a driver instance via the registry's binding table — see Phase 6). No giant union type. | Composition over enum. |
| **D10** | `ProviderDefinition.resolveProtocol()` and BAML-specific provider definition fields are removed (or left in place but unread). Provider definitions for native-driven providers (Fireworks) declare `driverId: 'openai-chat-completions'` and `codecId: 'native-chat-completions'`. The resolver looks these up in a binding table. | Configuration data, not type union. |
| **D11** | Legacy AppEvent variants (`thinking_chunk`, `thinking_end`, `message_chunk`, `message_start`, `message_end`, `tool_event` *(as a wrapper around xml-act `TurnEngineEvent`)*, `lens_start/chunk/end`, `raw_response_chunk`) are removed from the AppEvent union. Replaced by canonical TurnPart-based event interfaces (`thought_start/delta/end`, `assistant_message_start/delta/end`, `tool_call_start/input_delta/end`, `turn_usage`, `turn_finish`) plus tool-execution events (`tool_execution_started`, `tool_execution_ended`, `tool_observation`, `tool_error`). All defined as plain TypeScript interfaces with `type` discriminator and `forkId: string \| null` — matching the existing pattern in `events.ts`. | One canonical vocabulary. xml-act port, if undertaken later, would emit the new vocabulary. |
| **D12** | `CanonicalTurnProjection`, `CanonicalXmlProjection`, `ReplayProjection` are unhooked from the projection registry. Source files stay; they aren't read or fed events on the live path. | Orphan, don't delete. xml-act port could revive later. |
| **D13** | Native system prompt is rendered fresh by `renderNativeSystemPrompt(agentDef, skills) → SessionContextMessage`. Existing `renderSystemPrompt` (xml-act protocol) is unused on native; left in place as orphan. | Native doesn't need protocol prompts. Tools declared via wire `tools: [...]`. |
| **D14** | Tool ID format `call-{ord}-{ts36}` is generated inside the codec on `ToolCallStart`. Server-provided IDs are ignored. | Verified empirically (Fireworks). Single ID space. |
| **D15** | `TurnPartEvent` includes `MessageStart`/`MessageDelta`/`MessageEnd` for `delta.content` text. These events stream to UI and **DO** roll up into `parts: TurnPart[]` as a `MessagePart { id, text }`. Re-encoded on subsequent turns as `{ role: 'assistant', content: <text> }` (combined with thoughts and tool calls in the same wire-level assistant message). | Assistant text is part of the conversation — without persistence, the model wouldn't see its own prior responses, breaking multi-turn coherence. |
| **D16** | `ThoughtPart.level` is fixed at `'medium'` for now. Wire to per-agent config later. | Decoupled from the immediate need. Spec §7.1. |
| **D17** | Match the codebase's actual Effect API usage: `Schema.Class<T>()('Name', {...})` for data classes, `Schema.TaggedError<T>()('Name', {...})` (double parens) for errors, `Schema.Literal('a', 'b', 'c')` (variadic) for literal unions, `Schema.brand('BrandName')` (via `Schema.String.pipe(Schema.brand(...))`), `Stream.Stream<A, E, R>`, `Effect.Effect<A, E, R>`, `Context.Tag('Name')<Self, Shape>()`, `Layer.succeed`/`Layer.scoped`. **AppEvent variants are plain TypeScript interfaces with `type` discriminator** (matching the existing `events.ts` pattern), NOT Schema classes. | Consistency with the actual codebase, verified by grep. |
| **D18** | Compaction worker is **stubbed to no-op** for this plan. The native compaction pipeline is its own follow-up project. | Compaction depends on a BAML function (`CodingAgentCompact`) that's now orphaned. Stubbing avoids blocking on a separate large refactor. |
| **D19** | Phase ordering: scaffold → driver → codec → event vocabulary → projections → engine+dispatcher+cortex → wiring+e2e. Each phase's outputs verified before the next starts. | Smallest reversible steps. Each phase has clean unit-test bar. |

---

## 5. Phase 0 — Scaffold packages & contracts

**Goal.** Create the new packages with all type declarations and interfaces from the spec. No implementations. Typecheck passes; existing packages unchanged.

### 5.1 New packages

#### `packages/drivers/`

```
packages/drivers/
  package.json
  tsconfig.json
  src/
    driver.ts                   # Driver<W, C> interface; DriverCallOptions
    wire/
      chat-completions.ts       # ChatCompletionsRequest/ChatMessage/ChatTool/ChatToolCall types + ChatCompletionsStreamChunk Schema.Class
      completions.ts            # CompletionsRequest + CompletionsStreamChunk (for completions paradigm later)
    errors.ts                   # DriverError tagged error
    index.ts
```

`package.json`:
- name: `@magnitudedev/drivers`
- deps: `effect`
- peerDeps: `@effect/platform` (for HttpClient — Phase 1)

#### `packages/codecs/`

```
packages/codecs/
  package.json
  tsconfig.json
  src/
    memory/
      ids.ts                    # newToolCallId, newThoughtId, newMessageId — plain string constructors
      content-part.ts           # re-export from @magnitudedev/tools
      message.ts                # Message Schema.Union: SessionContext, ForkContext, Compacted, AssistantTurn, Inbox
      turn-part.ts              # TurnPart Schema.Union: ThoughtPart, ToolCallPart
      result-entry.ts           # ResultEntry Schema.Union (with toolCallId on tool variants)
      timeline-entry.ts         # TimelineEntry Schema.Union (unchanged from existing)
    events/
      turn-part-event.ts        # TurnPartEvent Schema.Union: ThoughtStart/Delta/End, MessageStart/Delta/End, ToolCallStart/InputDelta/End, TurnUsage, TurnFinish
    tools/
      tool-def.ts               # ToolDef Schema.Class
    codec.ts                    # Codec<W, C> interface; EncodeOptions; CodecEncodeError, CodecDecodeError
    adapters/
      model-adapter.ts          # ModelAdapter interface (for completions paradigm — declared, no impls)
    impls/
      .gitkeep                  # native impl in Phase 2
    index.ts
```

`package.json`:
- name: `@magnitudedev/codecs`
- deps: `effect`, `@magnitudedev/drivers`, `@magnitudedev/tools`

### 5.2 Files & contents (Phase 0)

#### `packages/codecs/src/memory/ids.ts`

**Branded IDs are not introduced.** Earlier plan drafts proposed `Schema.brand`-ed `ForkId`, `TurnId`, `ToolCallId`, `ThoughtId`, `MessageId`, `ModelId`. These would have been runtime-checked at codec boundaries. Dropped because:
- Existing code uses plain `string` for fork/turn/chain IDs (verified in `packages/agent/src/events.ts` and projections).
- Brands add type-import burden without runtime benefit (these IDs are never untrusted input).
- The codec's job is encode/decode of LLM wire messages — not validating internal IDs.

All IDs are typed as `string`. Constructors (`newToolCallId(ord)`, `newThoughtId()`, `newMessageId()`) live in `packages/codecs/src/memory/ids.ts` (Phase 0) and are the only producers.

`ModelId` similarly stays as the existing branded type from `packages/providers/src/types.ts` (already defined by `providers`); we don't redefine it.

#### `packages/codecs/src/memory/content-part.ts`

Re-exports `ContentPart`, `ImageMediaType` from `@magnitudedev/tools`. No definitions here.

#### `TurnPart` definition

Defined as a **plain TS discriminated union** in `packages/agent/src/projections/memory.ts` (where it's primarily consumed). No Schema; producers construct with statically-known shapes.

```ts
export type TurnPart =
  | { readonly type: 'thought';   readonly id: string; readonly level: 'low'|'medium'|'high'; readonly text: string }
  | { readonly type: 'message';   readonly id: string; readonly text: string }
  | { readonly type: 'tool_call'; readonly id: string; readonly toolName: string; readonly input: unknown }
```

The codec imports `TurnPart` as a type-only import from `@magnitudedev/agent` (paths align with the package's existing src tree). No new sub-package. See §5.5 for dep-graph note.

`MessagePart` (the `'message'` variant) carries the assistant's user-facing text response. It is encoded into the `content: string` field of the wire-level assistant message. See D15.

**Branded ID strings.** The plan earlier defined branded `ThoughtId`/`ToolCallId`/`MessageId` Schema brands. We **drop those brands** — they were unused at type level (the values are produced internally and consumed internally). All ID fields are `string`. Validity is enforced by construction (`newThoughtId`, `newToolCallId`, `newMessageId` helpers in the codec).

#### Memory `Message` type — plain TS, retained shape, evolved fields

**Decision (revised in this iteration).** The existing `Message` union in `packages/agent/src/projections/memory.ts` is a plain TS discriminated union using `type` as the discriminator (not `_tag`). It is consumed by every memory reader, projection, and renderer. **Do not migrate to `Schema.TaggedClass`** — that would be a paradigm change with massive ripple, with no runtime benefit (the type isn't validated at any boundary; producers construct it with statically-known shapes).

**Approach.** Keep `Message` as a plain TS union with `type` discriminator. Change only the `assistant_turn` variant to carry `parts: TurnPart[]` instead of `content: ContentPart[]`:

```ts
// packages/agent/src/projections/memory.ts (existing file, modified)
export type Message =
  | { readonly type: 'session_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | {
      readonly type: 'assistant_turn'
      readonly source: 'agent'
      readonly turnId:     string
      readonly parts:      TurnPart[]      // CHANGED — was: content: ContentPart[]
      readonly strategyId: StrategyId
    }
  | { readonly type: 'compacted';    readonly source: 'system'; readonly content: ContentPart[] }
  | { readonly type: 'fork_context'; readonly source: 'system'; readonly content: ContentPart[] }
  | {
      readonly type: 'inbox'
      readonly source: 'system'
      readonly results:  readonly ResultEntry[]
      readonly outcomes: readonly ResultEntry[]
      readonly timeline: readonly TimelineEntry[]
    }
```

`TurnPart` is a plain TS union (also no Schema):

```ts
// packages/agent/src/projections/memory.ts (or co-located util)
export type TurnPart =
  | { readonly type: 'thought';   readonly id: string; readonly level: 'low'|'medium'|'high'; readonly text: string }
  | { readonly type: 'message';   readonly id: string; readonly text: string }
  | { readonly type: 'tool_call'; readonly id: string; readonly toolName: string; readonly input: unknown }
```

**Why no Schema?** Schema is for validating data crossing trust boundaries (wire input, user input). `Message` is internal projection state — producers construct it with the right shape; no validation needed. We avoid Schema overhead and avoid the `_tag` vs `type` discriminator clash with the existing 60+ consumers.

The codec imports `Message` and `TurnPart` as **types** from `@magnitudedev/agent/projections/memory` (or a co-located types module to avoid pulling in the projection runtime). See §11 dep-graph note.

#### Inbox types — stay in `packages/agent/src/inbox/types.ts`

**Decision (revised in this iteration).** The existing inbox types in `packages/agent/src/inbox/types.ts` are plain TypeScript types using `kind` as the discriminator (not `_tag`). They are consumed by many existing renderers, tests, and CLI components. Migrating them to `Schema.TaggedClass` (with `_tag`) would cause a massive ripple of breakage across orphan and live code.

**Approach.** Keep these types where they are, as plain TS. The codec imports them as **types only** (`import type`) and consumes them in `encode`. No Schema wrapping needed because:
- They're input to the codec (memory→wire), never wire→memory.
- They're not validated at runtime — the producers (memory projection handlers) construct them with statically-known shapes.

**Two changes** to the existing types in Phase 4:
1. Add `toolCallId: string` to `ToolObservationResultItem` and `ToolErrorResultItem`.
2. Remove `query: string | null` from `ToolObservationResultItem` (filter system gone — D4 in §4).

The codec `packages/codecs/src/index.ts` re-exports the relevant inbox type names for ergonomic import:
```ts
export type {
  ResultEntry, TimelineEntry, TurnResultItem,
  ToolObservationResultItem, ToolErrorResultItem,
  /* ... */
} from '@magnitudedev/agent/inbox/types'
```

This creates a circular dep risk (`@magnitudedev/codecs` → `@magnitudedev/agent`). To avoid: split inbox types into a tiny new package `@magnitudedev/inbox-types` with no deps, and have both `agent` and `codecs` import from it. **Decision: do this split in Phase 4** to keep the dep graph clean. Files to move (no logic change, type-only):
- `packages/agent/src/inbox/types.ts` → `packages/inbox-types/src/index.ts`
- `packages/agent/src/content.ts` → `packages/inbox-types/src/content.ts` (ContentPart type)
- `@magnitudedev/agent/inbox/types` becomes `export * from '@magnitudedev/inbox-types'` (a re-export shim — agent code keeps importing from its old path).

#### `packages/codecs/src/events/turn-part-event.ts` — plain TS union

`TurnPartEvent` is the codec's output type from `decode`. The codec produces it; the agent consumes it. No untrusted boundary, no Schema. Plain TS discriminated union with `type` discriminator (matches event-core convention):

```ts
export type TurnPartEvent =
  | { readonly type: 'thought_start';        readonly id: string; readonly level: 'low'|'medium'|'high' }
  | { readonly type: 'thought_delta';        readonly id: string; readonly text: string }
  | { readonly type: 'thought_end';          readonly id: string }
  | { readonly type: 'message_start';        readonly id: string }
  | { readonly type: 'message_delta';        readonly id: string; readonly text: string }
  | { readonly type: 'message_end';          readonly id: string }
  | { readonly type: 'tool_call_start';      readonly id: string; readonly toolName: string }
  | { readonly type: 'tool_call_input_delta'; readonly id: string; readonly jsonChunk: string }
  | { readonly type: 'tool_call_end';        readonly id: string; readonly input: unknown }
  | { readonly type: 'turn_usage';           readonly inputTokens: number; readonly outputTokens: number; readonly cacheReadTokens: number | null; readonly cacheWriteTokens: number | null }
  | { readonly type: 'turn_finish';          readonly reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other' }
```

The codec emits `{ type: 'thought_start', ... }` literal objects from `processChunk`. No `new ThoughtStart(...)` constructors.

The agent's `liftTurnPartEvent` (Phase 3) consumes by `event.type` and produces the corresponding agent `AppEvent` with added `forkId`/`turnId`. Codec-side and agent-side `type` discriminators are aligned where they map directly:

| Codec `type` | Agent AppEvent `type` |
|---|---|
| `thought_start`         | `thought_start` |
| `thought_delta`         | `thought_delta` |
| `thought_end`           | `thought_end` |
| `message_start`         | `assistant_message_start` |
| `message_delta`         | `assistant_message_delta` |
| `message_end`           | `assistant_message_end` |
| `tool_call_start`       | `tool_call_start` |
| `tool_call_input_delta` | `tool_call_input_delta` |
| `tool_call_end`         | `tool_call_end` |
| `turn_usage`            | `turn_usage` |
| `turn_finish`           | `turn_finish` |

The `message_*` rename to `assistant_message_*` on the agent side avoids a name clash with the existing user-message-related events. The lift function does that rename.

#### `packages/codecs/src/tools/tool-def.ts`

```ts
export class ToolDef extends Schema.Class<ToolDef>('ToolDef')({
  name:        Schema.String,
  description: Schema.String,
  parameters:  Schema.Unknown,   // JSON Schema object
}) {}
```

#### `packages/codecs/src/codec.ts`

```ts
export interface EncodeOptions {
  readonly thinkingLevel?: 'low' | 'medium' | 'high'
  readonly maxTokens?:     number
  readonly stopSequences?: readonly string[]
}

export class CodecEncodeError extends Schema.TaggedError<CodecEncodeError>()(
  'CodecEncodeError', { reason: Schema.String, context: Schema.Unknown },
) {}
export class CodecDecodeError extends Schema.TaggedError<CodecDecodeError>()(
  'CodecDecodeError', { reason: Schema.String, partial: Schema.Unknown },
) {}

export interface Codec<WireRequest, WireChunk> {
  readonly id: string
  readonly encode: (
    memory: readonly Message[],
    tools:  readonly ToolDef[],
    options: EncodeOptions,
  ) => Effect.Effect<WireRequest, CodecEncodeError>

  readonly decode: (
    chunks: Stream.Stream<WireChunk, DriverError>,
  ) => Stream.Stream<TurnPartEvent, CodecDecodeError | DriverError>
}
```

#### `packages/codecs/src/adapters/model-adapter.ts`

```ts
export interface ModelAdapter {
  readonly id: string
  readonly encodeTools:  (tools: readonly ToolDef[]) => string
  readonly encodePrompt: (memory: readonly Message[], tools: readonly ToolDef[], options: EncodeOptions) => string
  readonly decode:       (text: Stream.Stream<string, DriverError>) => Stream.Stream<TurnPartEvent, CodecDecodeError | DriverError>
}
```

(Declared only; no impls in this plan.)

#### `packages/drivers/src/wire/chat-completions.ts`

`ChatCompletionsRequest` (interface), `ChatMessage` union, `ChatTool` (interface), `ChatToolCall` (interface), `ChatCompletionsStreamChunk` (`Schema.Class` with full validation of OpenAI/Fireworks SSE chunk shape).

#### `packages/drivers/src/wire/completions.ts`

`CompletionsRequest` (interface), `CompletionsStreamChunk` (`Schema.Class`). Declared for future completions paradigm.

#### `packages/drivers/src/driver.ts`

```ts
import * as HttpClient from '@effect/platform/HttpClient'
import { Effect, Stream } from 'effect'
import { DriverError } from './errors'

export interface DriverCallOptions {
  readonly endpoint:  string
  readonly authToken: string
}

export interface Driver<WireRequest, WireChunk> {
  readonly id: string
  readonly send: (
    request: WireRequest,
    options: DriverCallOptions,
  ) => Effect.Effect<
    Stream.Stream<WireChunk, DriverError>,
    DriverError,
    HttpClient.HttpClient
  >
}
```

`HttpClient.HttpClient` from `@effect/platform` (satisfied by `FetchHttpClient.layer` from `@effect/platform` at the app root). **No `TraceEmitter` requirement** — driver tracing is a follow-up (extending `@magnitudedev/tracing`'s `TraceInput` union is out of scope for this plan).

**Cancellation/timeouts.** No `signal` or `timeoutMs` fields. Effect interruption is the standard cancellation path; callers wrap `Driver.send(...)` in `Effect.timeout` / `Effect.race` for time-based limits and rely on `Effect.interrupt` for user-initiated cancel.

#### `packages/drivers/src/errors.ts`

```ts
export class DriverError extends Schema.TaggedError<DriverError>()(
  'DriverError',
  {
    reason:  Schema.String,
    status:  Schema.NullOr(Schema.Number),
    body:    Schema.Unknown,
  },
) {}
```

### 5.3 Workspace integration

- Update root `package.json` workspaces if needed (already includes `packages/*`).
- Run `bun install` to link packages.
- Add `tsconfig.json` per package (template: `packages/providers/tsconfig.json`).
- `packages/codecs/package.json` has dep on `@magnitudedev/drivers` workspace package.

### 5.4 Verification

- `bun install` succeeds.
- `bunx tsc --noEmit` (or per-package equivalent) passes for `packages/drivers/` and `packages/codecs/`.
- Existing packages still compile (we haven't touched them yet).
- No runtime tests in this phase — nothing to run yet.

### 5.5 Files created (summary)

```
packages/drivers/package.json
packages/drivers/tsconfig.json
packages/drivers/src/index.ts
packages/drivers/src/driver.ts
packages/drivers/src/errors.ts
packages/drivers/src/wire/chat-completions.ts
packages/drivers/src/wire/completions.ts

packages/codecs/package.json
packages/codecs/tsconfig.json
packages/codecs/src/index.ts
packages/codecs/src/codec.ts
packages/codecs/src/memory/ids.ts
packages/codecs/src/memory/content-part.ts
packages/codecs/src/memory/turn-part.ts
packages/codecs/src/memory/message.ts
packages/codecs/src/memory/result-entry.ts
packages/codecs/src/memory/timeline-entry.ts
packages/codecs/src/events/turn-part-event.ts
packages/codecs/src/tools/tool-def.ts
packages/codecs/src/adapters/model-adapter.ts
packages/codecs/src/impls/.gitkeep
```

---

## 6. Phase 1 — `OpenAIChatCompletionsDriver`

**Goal.** Implement the driver as a standalone Effect service with full unit-test coverage and a live integration test against Fireworks.

### 6.1 Files

```
packages/drivers/src/impls/
  openai-chat-completions.ts    # the driver value
  sse.ts                        # SSE line buffer + chunk parser
packages/drivers/src/__tests__/
  sse.test.ts
  openai-chat-completions.test.ts
  live-fireworks.test.ts        # gated by RUN_LIVE_TESTS=1
  fixtures/
    sse-basic.txt
    sse-split-mid-line.txt
    sse-tool-calls.txt
    sse-reasoning.txt
```

### 6.2 SSE helper (`sse.ts`)

Pure transformer. Signature:

```ts
export const sseChunks: (
  bytes: Stream.Stream<Uint8Array, DriverError>,
) => Stream.Stream<unknown, DriverError>
```

Behavior:

- Decode bytes as UTF-8 incrementally (use `TextDecoder({ stream: true })`).
- Maintain a line buffer (`string`).
- Split on `\n`; treat lines starting with `data: ` as event payloads.
- Empty line (`\n\n` boundary) flushes the current event.
- Lines starting with `:` are comments — skip.
- Payload `[DONE]` ends the stream.
- Each non-DONE payload is `JSON.parse`d; emit the parsed object.
- Malformed JSON → fail with `DriverError { reason: 'sse_parse_failed', body: rawLine }`.
- Stream end before `[DONE]` is acceptable (server may close cleanly).

### 6.3 Driver impl (`openai-chat-completions.ts`)

Uses `@effect/platform`'s `HttpClient` abstraction. The runtime layer (`@effect/platform`'s `FetchHttpClient.layer`) supplies the `HttpClient.HttpClient` Tag at the application root. The driver consumes the Tag via `yield* HttpClient.HttpClient`. (`FetchHttpClient` works on Bun since Bun ships native `fetch`.)

```ts
import { Effect, Schema, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import * as HttpClientRequest from '@effect/platform/HttpClientRequest'
import * as HttpClientResponse from '@effect/platform/HttpClientResponse'
import { sseChunks } from './sse'
import { ChatCompletionsStreamChunk, type ChatCompletionsRequest } from '../wire/chat-completions'
import { DriverError } from '../errors'
import type { Driver, DriverCallOptions } from '../driver'

export const OpenAIChatCompletionsDriver: Driver<ChatCompletionsRequest, ChatCompletionsStreamChunk> = {
  id: 'openai-chat-completions',

  send: (request, options: DriverCallOptions) => Effect.gen(function* () {
    const httpClient = yield* HttpClient.HttpClient

    // Build request. `bodyJson` is a combinator that returns an Effect because body
    // construction itself can fail (e.g. unserializable values).
    //
    // `HttpClientRequest.post(url)` → HttpClientRequest (sync). We then:
    //   1. setHeader (sync, returns HttpClientRequest)
    //   2. setHeader (sync)
    //   3. bodyJson(request)(req) → Effect<HttpClientRequest, BodyError>
    // The pipe chain mixes sync and Effect-returning combinators; we use `Effect.flatMap` for
    // the bodyJson step explicitly.
    const httpRequest = yield* HttpClientRequest.post(`${options.endpoint}/chat/completions`).pipe(
      HttpClientRequest.setHeader('Authorization', `Bearer ${options.authToken}`),
      HttpClientRequest.setHeader('Accept',         'text/event-stream'),
      // bodyJson expects to be applied as `bodyJson(body)(request)` and returns Effect.
      // We chain by piping through an Effect.succeed first, then flatMap.
      req => HttpClientRequest.bodyJson(request)(req),
      Effect.mapError(err => new DriverError({
        reason: 'request_build_failed',
        status: null,
        body:   { message: String(err) },
      })),
    )

    // Execute and check status. HttpClient.execute respects Effect interruption (cancels in-flight request).
    const responseEffect = httpClient.execute(httpRequest).pipe(
      Effect.mapError(err => new DriverError({
        reason: 'http_failed',
        status: null,
        body:   { message: err instanceof Error ? err.message : String(err) },
      })),
      Effect.flatMap(response => {
        if (response.status >= 400) {
          return response.text.pipe(
            Effect.orElseSucceed(() => ''),
            Effect.flatMap(bodyText =>
              Effect.fail(new DriverError({
                reason: 'http_status',
                status: response.status,
                body:   bodyText,
              })),
            ),
          )
        }
        return Effect.succeed(response)
      }),
    )

    // HttpClientResponse.stream is a module-level helper: takes Effect<Response,…> and returns Stream<Uint8Array,…>.
    // It runs the effect, lifts the response body into a Stream, and propagates response errors.
    const byteStream: Stream.Stream<Uint8Array, DriverError, never> =
      HttpClientResponse.stream(responseEffect).pipe(
        Stream.mapError(err =>
          err instanceof DriverError
            ? err
            : new DriverError({
                reason: 'transport_failed',
                status: null,
                body:   { message: err instanceof Error ? err.message : String(err) },
              }),
        ),
      )

    // bytes → parsed SSE JSON → typed chunk
    const chunkStream: Stream.Stream<ChatCompletionsStreamChunk, DriverError> = sseChunks(byteStream).pipe(
      Stream.mapEffect(json =>
        Schema.decodeUnknown(ChatCompletionsStreamChunk)(json).pipe(
          Effect.mapError(err => new DriverError({
            reason: 'chunk_decode_failed',
            status: null,
            body:   { error: String(err), json },
          })),
        ),
      ),
    )

    return chunkStream
  }),
}
```

(Tracing is **intentionally omitted** in this plan. Adding driver-emitted traces requires extending `TraceInput` in the upstream `@magnitudedev/tracing` package — out of scope. Cortex-level traces around `TurnEngine.runTurn` continue to work via the existing `withTraceScope` mechanism.)

Implementation notes:

- **`HttpClient.HttpClient` is provided by `@effect/platform`'s `FetchHttpClient.layer`** at the application root. The driver doesn't pick a specific platform — it just consumes the Tag. `FetchHttpClient.layer` works on Bun (and Node 18+) because both ship native `fetch`.
- **Effect interruption ↔ HTTP cancellation.** `httpClient.execute` honors Effect interruption: when the calling fiber is interrupted, the HTTP request is aborted and the response body stream is closed. No manual `AbortController` plumbing needed.
- **No `signal` or `timeoutMs` on `DriverCallOptions`.** Callers wrap `Driver.send(...)` in `Effect.timeout` / `Effect.race` for time-based cancellation; `Effect.interrupt` handles user-initiated cancellation (e.g., interrupt event from cortex).
- **`HttpClientRequest.bodyJson(request)`** — JSON-serializes and sets Content-Type. Returns `Effect<HttpClientRequest, BodyError>` (the body construction itself can fail on circular references etc.); we map any failure to `DriverError`.
- **`HttpClientResponse.stream(effect)` is a module-level helper**, not a property on the response object. It takes the response Effect and returns `Stream<Uint8Array, ResponseError | E>`. We compose the status-check inside the response Effect first, then lift to bytes via this helper.
- **Effect now `yield*`s the chunk stream**, not the raw response — the `Effect.gen` block returns `chunkStream` after composition; the response Effect is consumed inside `HttpClientResponse.stream`.
- **Tracing is out of scope.** The driver emits no traces. Adding driver traces requires modifying `@magnitudedev/tracing`'s `TraceInput` union (the type re-exported into `@magnitudedev/providers/resolver/tracing.ts:2`); that's a follow-up. Cortex-level tracing (around `TurnEngine.runTurn`) is unaffected — existing `withTraceScope` continues to work.

### 6.5 `Driver` interface (Phase 0 final shape)

`packages/drivers/src/driver.ts`:

```ts
import * as HttpClient from '@effect/platform/HttpClient'
import { TraceEmitter } from '@magnitudedev/providers'

export interface DriverCallOptions {
  readonly endpoint:  string
  readonly authToken: string
}

export interface Driver<WireRequest, WireChunk> {
  readonly id: string
  readonly send: (
    request: WireRequest,
    options: DriverCallOptions,
  ) => Effect.Effect<
    Stream.Stream<WireChunk, DriverError>,
    DriverError,
    HttpClient.HttpClient
  >
}
```

The `HttpClient.HttpClient` requirement is satisfied at the app root by `FetchHttpClient.layer` (from `@effect/platform`). **No `TraceEmitter` requirement** — driver tracing is a follow-up.

### 6.6 Wiring `FetchHttpClient.layer` into the runtime

**File:** `packages/agent/src/index.ts` (or wherever the live layer is composed)

```ts
import { FetchHttpClient } from '@effect/platform'
// ...
const HttpClientLive = FetchHttpClient.layer

// Provide to the agent's Live layer composition:
const Live = Layer.mergeAll(
  HttpClientLive,
  TurnEngineLive,
  ToolDispatcherLive,
  ProtocolBindingsLive,
  // ... other layers
)
```

For tests, substitute with a mock client via `Layer.succeed(HttpClient.HttpClient, mockClient)` or `HttpClient.layerMergedContext(...)`.

(`FetchHttpClient` works on Bun, Node 18+, and any modern runtime with global `fetch`. There is no `BunHttpClient` export in `@effect/platform-bun`; the earlier draft was wrong.)

### 6.5 Tests

| Test | What it verifies |
|---|---|
| `sse.test.ts` — single complete event | One `data: {...}` parses correctly. |
| `sse.test.ts` — split mid-line | Two byte chunks where the line break falls in the middle of a JSON payload — reassembles correctly. |
| `sse.test.ts` — `[DONE]` terminator | Stream ends cleanly on DONE. |
| `sse.test.ts` — malformed JSON | Fails with `DriverError`. |
| `sse.test.ts` — comment lines | Lines starting with `:` are skipped. |
| `openai-chat-completions.test.ts` — happy path | Mocked `HttpClient` returns canned SSE bytes (`fixtures/sse-basic.txt`); driver decodes into typed `ChatCompletionsStreamChunk`s. |
| `openai-chat-completions.test.ts` — auth failure | 401 response → `DriverError { status: 401 }`. |
| `openai-chat-completions.test.ts` — rate limit | 429 → `DriverError { status: 429 }`. |
| `openai-chat-completions.test.ts` — server error | 500 → `DriverError { status: 500 }`. |
| `openai-chat-completions.test.ts` — interruption | Effect-level interrupt cancels the in-flight request. |
| `openai-chat-completions.test.ts` — chunk decode failure | Server emits a chunk that doesn't match schema → `DriverError { reason: 'chunk_decode_failed' }`. |
| `live-fireworks.test.ts` (gated) | Real call to `https://api.fireworks.ai/inference/v1/chat/completions` with `accounts/fireworks/models/kimi-k2p6`, simple system+user prompt, no tools. Asserts stream emits chunks, decodes, terminates. Skipped unless `RUN_LIVE_TESTS=1 FIREWORKS_API_KEY=...` set. |

### 6.6 Verification

- `bunx --bun vitest run packages/drivers/src/__tests__` passes.
- `RUN_LIVE_TESTS=1 FIREWORKS_API_KEY=... bunx --bun vitest run packages/drivers/src/__tests__/live-fireworks.test.ts` passes.

### 6.7 Files created in Phase 1

```
packages/drivers/src/impls/openai-chat-completions.ts
packages/drivers/src/impls/sse.ts
packages/drivers/src/__tests__/sse.test.ts
packages/drivers/src/__tests__/openai-chat-completions.test.ts
packages/drivers/src/__tests__/live-fireworks.test.ts
packages/drivers/src/__tests__/fixtures/sse-basic.txt
packages/drivers/src/__tests__/fixtures/sse-split-mid-line.txt
packages/drivers/src/__tests__/fixtures/sse-tool-calls.txt
packages/drivers/src/__tests__/fixtures/sse-reasoning.txt
```

Files modified:
```
packages/drivers/src/driver.ts        (DriverCallOptions trimmed)
```

---

## 7. Phase 2 — `NativeChatCompletionsCodec`

**Goal.** Implement encode + decode. Pure, testable in isolation.

### 7.1 Files

```
packages/codecs/src/impls/native-chat-completions/
  index.ts                    # NativeChatCompletionsCodec value
  encode.ts                   # encode(memory, tools, options) → ChatCompletionsRequest
  encode-message.ts           # one Message → ChatMessage[]
  encode-tool-def.ts          # ToolDef → ChatTool
  decode.ts                   # decode(chunkStream) → Stream<TurnPartEvent>
  decode-state.ts             # per-call decoder state machine
  ids.ts                      # call-{ord}-{ts36} generator (extracted from xml-act helper)
packages/codecs/src/__tests__/native-chat-completions/
  encode.test.ts
  decode.test.ts
  end-to-end.test.ts          # captured Fireworks SSE fixtures
  fixtures/
    fireworks-thoughts-only.txt
    fireworks-tool-calls.txt
    fireworks-parallel-tool-calls.txt
    fireworks-multimodal-tool-result.txt
```

### 7.2 Encode

**`encode-message.ts`** — one `Message` to one or more `ChatMessage`s:

| Memory `Message` | → ChatMessage(s) |
|---|---|
| `SessionContextMessage { content }` | `{ role: 'system', content }` (or `'developer'` if model capability allows) |
| `ForkContextMessage { content }` | `{ role: 'system', content }` |
| `CompactedMessage { content }` | `{ role: 'system', content }` (prefixed `<compacted>...</compacted>` text part) |
| `AssistantTurnMessage { parts }` | One `{ role: 'assistant', content?, reasoning_content?, tool_calls? }`. `reasoning_content` = concatenation of `ThoughtPart.text`s separated by `\n\n` (omitted if model capability says no reasoning). `content` = concatenation of `MessagePart.text`s separated by `\n\n` (or omitted/null if no MessagePart). `tool_calls` = each `ToolCallPart` as `{ id: part.id, type: 'function', function: { name: part.toolName, arguments: JSON.stringify(part.input) } }` (omitted if no ToolCallParts). If `parts` is empty (defensive), emit `{ role: 'assistant', content: '' }`. |
| `InboxMessage { results, timeline }` | See encode-inbox below. |

**Inbox encoding** (`encode-message.ts`, `encodeInbox` helper):

For each `ResultEntry`:
- `TurnResultsEntry`:
  - For each `ToolObservationResultItem { toolCallId, content }`: emit `{ role: 'tool', tool_call_id: toolCallId, content }` (passing `ContentPart[]` through — verified Fireworks-compatible).
  - For each `ToolErrorResultItem { toolCallId, message }`: emit `{ role: 'tool', tool_call_id: toolCallId, content: message ?? '<error>' }`.
  - For `ToolParseErrorResultItem`, `StructuralParseErrorResultItem`: collect into a single `{ role: 'user', content: '<system>...parse errors rendered as text...</system>' }` message after the tool messages.
  - For `MessageAckResultItem`, `NoToolsOrMessagesResultItem`: same — collect into the trailing user note.
- `InterruptedEntry`, `ErrorEntry`, `OneshotLivenessEntry`, `YieldWorkerRetriggerEntry`, `NoopEntry`: render as text in the trailing user note.

For `TimelineEntry`:
- Render to text via existing `formatTimeline` logic (preserve current xml-act-side rendering for content like `<user-message>`, `<presence>`, etc.) — *but* output goes into a single multimodal `{ role: 'user', content: ContentPart[] }` message. Image attachments (e.g., from `ObservationEntry`) become image `ContentPart`s.

The trailing user note + the timeline user message may be merged into one user message at the end, or kept as two — **decision: merge into one**. After all tool messages, append one `{ role: 'user', content: ContentPart[] }` containing the timeline + the trailing system note rendered as text + image parts.

**`encode-tool-def.ts`** — `ToolDef → ChatTool`:

```ts
{ type: 'function', function: { name: td.name, description: td.description, parameters: td.parameters } }
```

`parameters` is already a JSON Schema object (the agent's `extract-tool-defs` step in Phase 5 produces it).

**`encode.ts`** top-level:

```ts
export const encode = (
  memory: readonly Message[],
  tools:  readonly ToolDef[],
  options: EncodeOptions,
  config: { wireModelName: string; defaultMaxTokens: number; supportsReasoning: boolean; supportsVision: boolean },
) => Effect.sync(() => {
  const messages: ChatMessage[] = memory.flatMap(m => encodeMessage(m, config))
  const chatTools = tools.length > 0 ? tools.map(encodeToolDef) : undefined

  return {
    model:        config.wireModelName,
    messages,
    tools:        chatTools,
    stream:       true,
    max_tokens:   options.maxTokens ?? config.defaultMaxTokens,
    stop:         options.stopSequences,
    // temperature, etc. as needed
  } satisfies ChatCompletionsRequest
})
```

`config` is sourced from the `BoundModel.wireConfig` plus capability flags from `BoundModel.providerModel`. The codec receives it via partial application when constructed inside `BoundModel`.

### 7.3 Decode

**`decode-state.ts`** — pure state container:

```ts
interface DecoderState {
  ordinal: number
  openThoughtId: string | null
  openMessageId: string | null
  openToolCalls: Map<number /* index */, { id: string; toolName: string; argsBuffer: string }>
}
```

**`decode.ts`** — pure accumulator over chunk stream:

```ts
export const decode = (
  chunks: Stream.Stream<ChatCompletionsStreamChunk, DriverError>,
): Stream.Stream<TurnPartEvent, CodecDecodeError | DriverError> => {
  const initial: DecoderState = {
    ordinal: 0,
    openThoughtId: null,
    openMessageId: null,
    openToolCalls: new Map(),  // immutable-style: rebuilt per step, not mutated
  }
  return chunks.pipe(
    // Stream.mapAccum threads pure state through. Each chunk yields [newState, events[]].
    Stream.mapAccum(initial, (state, chunk) => {
      const result = processChunk(chunk, state)  // pure: returns { state, events }
      return [result.state, result.events]
    }),
    // Flatten events[] into the stream
    Stream.flatMap(events => Stream.fromIterable(events)),
  )
}
```

`processChunk(chunk, state) → { state: DecoderState; events: TurnPartEvent[] }` is pure: it produces the new state and the events for this chunk by case analysis on `delta.reasoning_content`, `delta.content`, `delta.tool_calls`, and `finish_reason`. The state object is rebuilt (not mutated) — `openToolCalls` is a new Map per step.

If the upstream chunk stream ends without ever emitting `finish_reason`, `processChunk` will not have closed open thoughts/messages/tool calls. We accept this as a malformed-stream condition; `TurnEngine`'s caller (Cortex) sees no `TurnFinish` event and treats it as `UnexpectedError` via the codec/driver layer — but in practice OpenAI/Fireworks always emits a final chunk with `finish_reason`.

(An earlier draft used a mutable closure variable for state. That's unsafe in Effect because streams may be replayed or consumed concurrently. `Stream.mapAccum` is the safe pure-functional equivalent.)

**`processChunk(chunk, state)`** is a **pure** function returning a new state and the events emitted by this chunk.

```ts
export const processChunk = (
  chunk: ChatCompletionsStreamChunk,
  state: DecoderState,
): { state: DecoderState; events: TurnPartEvent[] } => {
  const events: TurnPartEvent[] = []
  let s: DecoderState = state                                        // we'll rebuild s as we go
  const choice = chunk.choices[0]
  if (!choice) return { state: s, events }
  const delta = choice.delta

  // 1. reasoning_content — open/extend thought
  if (delta.reasoning_content) {
    if (s.openThoughtId === null) {
      const id = newThoughtId()
      s = { ...s, openThoughtId: id }
      events.push({ type: 'thought_start', id, level: 'medium' })
    }
    events.push({ type: 'thought_delta', id: s.openThoughtId!, text: delta.reasoning_content })
  }

  // 2. content — open/extend message; close any open thought first
  if (delta.content) {
    if (s.openThoughtId !== null) {
      events.push({ type: 'thought_end', id: s.openThoughtId })
      s = { ...s, openThoughtId: null }
    }
    if (s.openMessageId === null) {
      const id = newMessageId()
      s = { ...s, openMessageId: id }
      events.push({ type: 'message_start', id })
    }
    events.push({ type: 'message_delta', id: s.openMessageId!, text: delta.content })
  }

  // 3. tool_calls — close any open thought/message; open/extend tool calls (per index)
  if (delta.tool_calls && delta.tool_calls.length > 0) {
    if (s.openThoughtId !== null) {
      events.push({ type: 'thought_end', id: s.openThoughtId })
      s = { ...s, openThoughtId: null }
    }
    if (s.openMessageId !== null) {
      events.push({ type: 'message_end', id: s.openMessageId })
      s = { ...s, openMessageId: null }
    }
    const calls = new Map(s.openToolCalls)
    let ord = s.ordinal
    for (const tc of delta.tool_calls) {
      let entry = calls.get(tc.index)
      if (!entry) {
        ord += 1
        const id = newToolCallId(ord)
        const toolName = tc.function?.name ?? ''
        entry = { id, toolName, argsBuffer: '' }
        calls.set(tc.index, entry)
        events.push({ type: 'tool_call_start', id, toolName })
      } else if (tc.function?.name && !entry.toolName) {
        // Some providers send name in a follow-up chunk
        entry = { ...entry, toolName: tc.function.name }
        calls.set(tc.index, entry)
      }
      if (tc.function?.arguments) {
        entry = { ...entry, argsBuffer: entry.argsBuffer + tc.function.arguments }
        calls.set(tc.index, entry)
        events.push({ type: 'tool_call_input_delta', id: entry.id, jsonChunk: tc.function.arguments })
      }
    }
    s = { ...s, openToolCalls: calls, ordinal: ord }
  }

  // 4. finish_reason — close everything still open; emit usage + turn_finish
  if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
    if (s.openThoughtId !== null) {
      events.push({ type: 'thought_end', id: s.openThoughtId })
      s = { ...s, openThoughtId: null }
    }
    if (s.openMessageId !== null) {
      events.push({ type: 'message_end', id: s.openMessageId })
      s = { ...s, openMessageId: null }
    }
    for (const entry of s.openToolCalls.values()) {
      const parsedInput = tryParseJson(entry.argsBuffer)
      events.push({ type: 'tool_call_end', id: entry.id, input: parsedInput })
    }
    s = { ...s, openToolCalls: new Map() }
    if (chunk.usage) {
      events.push({
        type: 'turn_usage',
        inputTokens:      chunk.usage.prompt_tokens ?? 0,
        outputTokens:     chunk.usage.completion_tokens ?? 0,
        cacheReadTokens:  chunk.usage.prompt_tokens_details?.cached_tokens ?? null,
        cacheWriteTokens: null,
      })
    }
    events.push({ type: 'turn_finish', reason: mapReason(choice.finish_reason) })
  }

  return { state: s, events }
}
```

`tryParseJson(s)` returns `JSON.parse(s)` on success or `{ _parseError: s }` on failure. The plan accepts that on failure, the dispatched tool call will be schema-rejected by `ToolDispatcher` and surface as a `ToolErrorResultItem` to the model on the next turn. (We do **not** emit a separate `CodecDecodeError` for malformed tool args — keeps the stream non-fatal.)

`mapReason` maps OpenAI's `'stop' | 'length' | 'tool_calls' | 'content_filter' | unknown` to the spec's literal union, defaulting to `'other'` for unknowns.

**Note on `delta.tool_calls[i].id`.** The OpenAI/Fireworks SSE schema includes a server-assigned `id` on the first chunk for each tool call. Per **D14** (verified empirically), the codec ignores server IDs and generates its own `call-{ord}-{ts36}` IDs. The single ID space is the codec's. Tests must verify this.

### 7.4 ID generation

`packages/codecs/src/impls/native-chat-completions/ids.ts`:

```ts
const newToolCallId = (ord: number): string =>
  `call-${ord}-${Date.now().toString(36)}`

let thoughtCounter = 0
const newThoughtId = (): ThoughtId =>
  ThoughtId.makeUnsafe(`thought-${++thoughtCounter}-${Date.now().toString(36)}`)

let messageCounter = 0
const newMessageId = (): MessageId =>
  MessageId.makeUnsafe(`msg-${++messageCounter}-${Date.now().toString(36)}`)
```

(Module-local counters acceptable for streaming use — IDs only need uniqueness within a session, not globally across processes.)

### 7.5 The codec value

```ts
// packages/codecs/src/impls/native-chat-completions/index.ts
export const NativeChatCompletionsCodec = (config: {
  wireModelName: string
  defaultMaxTokens: number
  supportsReasoning: boolean
  supportsVision: boolean
}): Codec<ChatCompletionsRequest, ChatCompletionsStreamChunk> => ({
  id: 'native-chat-completions',
  encode: (memory, tools, options) => encode(memory, tools, options, config),
  decode,
})
```

The codec is a **factory** — `NativeChatCompletionsCodec(config)` returns a Codec instance. `BoundModel` constructs it with the per-model config.

### 7.6 Tests

| Test | Coverage |
|---|---|
| `encode.test.ts` — system messages | Session/Fork/Compacted → `role: 'system'` with correct content. |
| `encode.test.ts` — assistant turn (thoughts only) | `parts: [thought, thought]` → `{ role: 'assistant', reasoning_content: '...', content: null }`. |
| `encode.test.ts` — assistant turn (tool calls) | `parts: [toolCall]` → `{ role: 'assistant', tool_calls: [...] }`. |
| `encode.test.ts` — assistant turn (mixed) | Thoughts + tool calls → both fields populated. |
| `encode.test.ts` — inbox tool observation | One result item → `{ role: 'tool', tool_call_id: ..., content: ContentPart[] }`. |
| `encode.test.ts` — inbox parallel tool results | Two results → two tool messages, IDs preserved. |
| `encode.test.ts` — inbox multimodal | Image content in tool result passes through correctly. |
| `encode.test.ts` — inbox timeline | User message + observation merged into one user message. |
| `encode.test.ts` — tools | `ToolDef[]` → `tools: [{type: 'function', function: ...}]`. |
| `encode.test.ts` — capability gate | `supportsReasoning: false` → no `reasoning_content` field. |
| `decode.test.ts` — pure thought stream | reasoning_content chunks → `ThoughtStart`, deltas, `ThoughtEnd` on finish. |
| `decode.test.ts` — pure message stream | content chunks → `MessageStart`, deltas, `MessageEnd`. |
| `decode.test.ts` — single tool call | tool_calls deltas → `ToolCallStart`, input deltas, `ToolCallEnd` with parsed input. |
| `decode.test.ts` — parallel tool calls | indices 0 and 1 → two ID streams, both terminate on finish. |
| `decode.test.ts` — thought→tool transition | reasoning_content followed by tool_calls → ThoughtEnd before ToolCallStart. |
| `decode.test.ts` — usage + finish | `TurnUsage` + `TurnFinish` emitted at end. |
| `decode.test.ts` — malformed args | bad JSON in arguments → `ToolCallEnd { input: { _parseError } }` + non-fatal `CodecDecodeError`. |
| `decode.test.ts` — finish reason mapping | `'stop'`/`'length'`/`'tool_calls'`/`'content_filter'`/unknown → mapped correctly. |
| `end-to-end.test.ts` — captured Fireworks fixtures | Replay actual SSE chunk sequences captured from Fireworks/Kimi K2.6 → assert event sequences. One fixture per scenario (thoughts only, tool calls, parallel tools, multimodal results). |

### 7.7 Capturing fixtures

A small script `$M/scripts/capture-fireworks-sse.ts` makes a real Fireworks call and saves the raw SSE bytes to a fixture file. Run once per scenario; commit the fixtures.

### 7.8 Verification

- `bunx --bun vitest run packages/codecs/src/__tests__` passes.
- E2E test against captured fixtures matches expected event sequences.
- Manual: pipe codec.decode + console.log against a real Fireworks call. Inspect events.

### 7.9 Files created in Phase 2

```
packages/codecs/src/impls/native-chat-completions/
  index.ts
  encode.ts
  encode-message.ts
  encode-tool-def.ts
  decode.ts
  decode-state.ts
  ids.ts
packages/codecs/src/__tests__/native-chat-completions/
  encode.test.ts
  decode.test.ts
  end-to-end.test.ts
  fixtures/*.txt
$M/scripts/capture-fireworks-sse.ts
```

---

## 8. Phase 3 — Canonical event vocabulary refactor

**Goal.** Replace xml-act-coupled AppEvent variants with the canonical `TurnPartEvent`-based vocabulary. After this phase, `events.ts` has no xml-act-shaped event names.

### 8.1 Current AppEvent vocabulary (xml-act-coupled — to remove)

From `packages/agent/src/events.ts` (research):

- `thinking_chunk`, `thinking_end` — xml-act parser artifact
- `lens_start`, `lens_chunk`, `lens_end` — named-lens system (dropped)
- `message_start`, `message_chunk`, `message_end` — xml-act `<message>` tag (dropped — message tag is gone, see §1)
- `raw_response_chunk` — raw model text tee — xml-act-only purpose
- `tool_event` — wrapper carrying xml-act `TurnEngineEvent` shapes (`ToolInputStarted`, `ToolInputReady`, `ToolExecutionStarted`, `ToolExecutionEnded`, `ToolObservation`, `ToolParseError`, `StructuralParseError`)

### 8.2 New AppEvent vocabulary

The AppEvent union is partitioned into three groups:

**Group A — Turn lifecycle (unchanged):**
- `turn_started { forkId, turnId, chainId, ... }`
- `turn_outcome { forkId, turnId, outcome }` — outcome shape simplified, see §10
- `interrupt`, `soft_interrupt`, `wake`, `oneshot_task` (turn triggers — unchanged)

**Group B — Assistant turn streaming (NEW canonical events, replacing xml-act-shaped ones):**
- `thought_start { forkId, turnId, id: ThoughtId, level }`
- `thought_delta { forkId, turnId, id: ThoughtId, text }`
- `thought_end { forkId, turnId, id: ThoughtId }`
- `assistant_message_start { forkId, turnId, id: MessageId }`
- `assistant_message_delta { forkId, turnId, id: MessageId, text }`
- `assistant_message_end { forkId, turnId, id: MessageId }`
- `tool_call_start { forkId, turnId, id: ToolCallId, toolName }`
- `tool_call_input_delta { forkId, turnId, id: ToolCallId, jsonChunk }`
- `tool_call_end { forkId, turnId, id: ToolCallId, input }`
- `turn_usage { forkId, turnId, inputTokens, outputTokens, ... }`
- `turn_finish { forkId, turnId, reason }`

**Group C — Tool execution (NEW canonical events, replacing the `tool_event` wrapper):**
- `tool_execution_started { forkId, turnId, toolCallId, toolName }`
- `tool_execution_ended { forkId, turnId, toolCallId, toolName, durationMs }`
- `tool_observation { forkId, turnId, toolCallId, toolName, content: ContentPart[] }`
- `tool_error { forkId, turnId, toolCallId, toolName, status: 'error'|'rejected'|'interrupted', message? }`

**Group D — Other (unchanged):**
- `user_message`, `agent_created`, `task_created`, `task_updated`, `agent_block`, `observations_captured`, etc.

### 8.3 Event interface definitions

**Pattern.** Existing `events.ts` defines events as plain TypeScript `interface`s with a `type: string` discriminator and `forkId: string | null`. We follow the existing pattern exactly — NOT `Schema.TaggedClass`.

`packages/agent/src/events.ts` (new event interfaces):

```ts
// Group B — Assistant turn streaming
export interface ThoughtStart {
  readonly type: 'thought_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string             // ThoughtId
  readonly level: 'low' | 'medium' | 'high'
}

export interface ThoughtDelta {
  readonly type: 'thought_delta'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly text: string
}

export interface ThoughtEnd {
  readonly type: 'thought_end'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
}

export interface AssistantMessageStart {
  readonly type: 'assistant_message_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string             // MessageId
}

export interface AssistantMessageDelta {
  readonly type: 'assistant_message_delta'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly text: string
}

export interface AssistantMessageEnd {
  readonly type: 'assistant_message_end'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
}

export interface ToolCallStartEvent {
  readonly type: 'tool_call_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string             // ToolCallId
  readonly toolName: string
}

export interface ToolCallInputDelta {
  readonly type: 'tool_call_input_delta'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly jsonChunk: string
}

export interface ToolCallEndEvent {
  readonly type: 'tool_call_end'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly input: unknown
}

export interface TurnUsage {
  readonly type: 'turn_usage'
  readonly forkId: string | null
  readonly turnId: string
  readonly inputTokens: number
  readonly outputTokens: number
  readonly cacheReadTokens: number | null
  readonly cacheWriteTokens: number | null
}

export interface TurnFinish {
  readonly type: 'turn_finish'
  readonly forkId: string | null
  readonly turnId: string
  readonly reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'
}

// Group C — Tool execution
export interface ToolExecutionStarted {
  readonly type: 'tool_execution_started'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolName: string
}

export interface ToolExecutionEnded {
  readonly type: 'tool_execution_ended'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolName: string
  readonly durationMs: number
}

export interface ToolObservationEvent {
  readonly type: 'tool_observation'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolName: string
  readonly content: ContentPart[]
}

export interface ToolErrorEvent {
  readonly type: 'tool_error'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolName: string
  readonly status: 'error' | 'rejected' | 'interrupted'
  readonly message?: string
}
```

`AppEvent` is the existing TS union of all event interfaces. Add the new variants; remove the old ones (§8.4). Keep the existing structure (`type CodingAgentEvent = TurnStarted | TurnOutcomeEvent | UserMessage | ...`).

**Note on naming.** Some new events conflict with existing imported types from xml-act (e.g. `ToolCallStart` is a `TurnPartEvent` variant from the codec). To avoid name clashes, agent-side AppEvent interfaces append `Event` suffix where ambiguous (`ToolCallStartEvent`, `ToolCallEndEvent`, `ToolObservationEvent`, `ToolErrorEvent`). The codec's TurnPartEvent variants (`ThoughtStart`, `MessageStart`, `ToolCallStart`, etc.) are imported under a namespace alias to disambiguate.

### 8.4 Removed events

These existing interface definitions are **deleted** from `events.ts`:

- `ThinkingChunk`, `ThinkingEnd`
- `LensStart`, `LensChunk`, `LensEnd`
- `MessageStart`, `MessageChunkEvent`, `MessageEnd` (replaced by `AssistantMessage*`)
- `RawResponseChunk`
- `ToolEvent` (the wrapper) — and the import of `TurnEngineEvent` from `@magnitudedev/xml-act`

The `ParseFailureEvent` re-export and `MessageDestination` type stay; the latter is now used by the `send_message_to_*` tools (out of scope here).

### 8.5 Event lift helper

A small helper that lifts a `TurnPartEvent` (codec output, schema class) to the corresponding agent `AppEvent` (interface), adding `forkId` + `turnId`:

```ts
// packages/agent/src/workers/cortex/lift-event.ts (new)
import * as TPE from '@magnitudedev/codecs/events/turn-part-event'
import type {
  ThoughtStart, ThoughtDelta, ThoughtEnd,
  AssistantMessageStart, AssistantMessageDelta, AssistantMessageEnd,
  ToolCallStartEvent, ToolCallInputDelta, ToolCallEndEvent,
  TurnUsage, TurnFinish,
  CodingAgentEvent,
} from '../../events'

export const liftTurnPartEvent = (
  event: TPE.TurnPartEvent,
  forkId: string | null,
  turnId: string,
): CodingAgentEvent => {
  switch (event._tag) {
    case 'thought_start':         return { type: 'thought_start', forkId, turnId, id: event.id, level: event.level }
    case 'thought_delta':         return { type: 'thought_delta', forkId, turnId, id: event.id, text: event.text }
    case 'thought_end':           return { type: 'thought_end', forkId, turnId, id: event.id }
    case 'message_start':         return { type: 'assistant_message_start', forkId, turnId, id: event.id }
    case 'message_delta':         return { type: 'assistant_message_delta', forkId, turnId, id: event.id, text: event.text }
    case 'message_end':           return { type: 'assistant_message_end', forkId, turnId, id: event.id }
    case 'tool_call_start':       return { type: 'tool_call_start', forkId, turnId, id: event.id, toolName: event.toolName }
    case 'tool_call_input_delta': return { type: 'tool_call_input_delta', forkId, turnId, id: event.id, jsonChunk: event.jsonChunk }
    case 'tool_call_end':         return { type: 'tool_call_end', forkId, turnId, id: event.id, input: event.input }
    case 'turn_usage':            return { type: 'turn_usage', forkId, turnId, inputTokens: event.inputTokens, outputTokens: event.outputTokens, cacheReadTokens: event.cacheReadTokens, cacheWriteTokens: event.cacheWriteTokens }
    case 'turn_finish':           return { type: 'turn_finish', forkId, turnId, reason: event.reason }
  }
}
```

This is a thin add-context shim — codec emits paradigm-pure events; agent enriches with the orchestration context (forkId, turnId) it owns.

### 8.6 Files modified in Phase 3

```
packages/agent/src/events.ts               # delete xml-act event classes; add Group B + C event classes
packages/agent/src/workers/cortex/lift-event.ts   # NEW
```

Files implicitly affected (handled in Phase 4):
- `packages/agent/src/projections/memory.ts`
- `packages/agent/src/projections/display.ts`
- `packages/agent/src/projections/turn.ts`
- `packages/agent/src/projections/canonical-turn.ts` (orphaned)
- `packages/agent/src/projections/canonical-xml.ts` (orphaned)
- `packages/agent/src/projections/replay.ts` (orphaned)
- All tests that reference removed event names

### 8.7 Verification

- `bun typecheck` for `packages/agent` will now have many errors in projections/tests — expected, fixed in Phase 4.
- The events module compiles standalone.
- `lift-event` is unit-tested: each `TurnPartEvent` variant maps to the correct AppEvent (1-to-1).

---

## 9. Phase 4 — Memory & projection refactor

**Goal.** Update `MemoryProjection`, `DisplayProjection`, and inbox types to consume the new event vocabulary. Orphan xml-act-coupled projections.

### 9.1 `inbox/types.ts` — re-export from `@magnitudedev/codecs`

**File:** `packages/agent/src/inbox/types.ts`

Replace local definitions with re-exports from `@magnitudedev/codecs/memory/result-entry` and `.../timeline-entry`. Keep `inbox/compose.ts` as constructor helpers wrapping the imported classes.

The codec package's `ResultEntry` definitions already include `toolCallId` on `ToolObservationResultItem` and `ToolErrorResultItem` (Phase 0).

### 9.2 `inbox/compose.ts` — constructor helpers updated

```ts
// before:
export const toolObservation = (tagName: string, query: string | null, content: ContentPart[])
// after:
export const toolObservation = (toolCallId: ToolCallId, tagName: string, query: string | null, content: ContentPart[])
```

`tagName` is the tool name (kept for legacy display rendering and inbox identity). On native, `tagName === toolName`.

`query` is **removed** entirely. The filter/query system is gone; tools that need filtering accept it as part of `input`.

Updated signature:
```ts
export const toolObservation = (toolCallId: ToolCallId, tagName: string, content: ContentPart[])
```

Same for `toolError`.

### 9.3 `inbox/render.ts` and `render-results.ts` — update for shape

These files render inbox entries to text for two purposes:
1. xml-act wire format (orphaned).
2. CLI history display (live).

For (2), the renderer reads `ResultEntry.items[i]`. The new `toolCallId` field is ignored by render — rendering is by tagName + content. Verify by re-reading the code.

The `query` field removal: search for usages, remove.

### 9.4 `events.ts` — already done in Phase 3

### 9.5 `MemoryProjection` refactor

**File:** `packages/agent/src/projections/memory.ts`

Changes:

#### 9.5.1 `Message` type — `assistant_turn` carries `parts`

Replace local `Message` union with re-export from `@magnitudedev/codecs/memory/message`:

```ts
export {
  Message, SessionContextMessage, ForkContextMessage,
  CompactedMessage, AssistantTurnMessage, InboxMessage,
} from '@magnitudedev/codecs'
```

Local helpers like `textParts(...)` retained for use in non-assistant messages.

#### 9.5.2 New event handlers

The projection's `eventHandlers` gains:

- `thought_start { id, level }` — start a new `ThoughtPart`; track in working state.
- `thought_delta { id, text }` — append text to the open thought.
- `thought_end { id }` — close the thought; push as `ThoughtPart` to `working.parts`.
- `tool_call_start { id, toolName }` — start a `ToolCallPart` in working state.
- `tool_call_input_delta { id, jsonChunk }` — accumulate args for the tool call.
- `tool_call_end { id, input }` — close the tool call; push as `ToolCallPart` to `working.parts`.
- `turn_started` — reset working state.
- `turn_outcome` — finalize: push `AssistantTurnMessage { turnId, parts: working.parts, strategyId: 'native' }` to `fork.messages`.
- `tool_observation { toolCallId, content }` — append `ToolObservationResultItem` to pending inbox results.
- `tool_error { toolCallId, status, message }` — append `ToolErrorResultItem`.
- `assistant_message_start { id }` — start a `MessagePart` in working state.
- `assistant_message_delta { id, text }` — append to the open message text.
- `assistant_message_end { id }` — close; push as `MessagePart { id, text }` to `working.parts`. **Persisted** (D15).

#### 9.5.3 Working state per fork

```ts
interface ForkWorkingTurn {
  turnId: string | null
  parts: TurnPart[]
  openThought:    { id: string; level: 'low'|'medium'|'high'; text: string } | null
  openMessage:    { id: string; text: string } | null
  openToolCalls:  Map<string /* ToolCallId */, { toolName: string; argsBuffer: string }>
}
```

Notes:
- Multiple **tool calls** can be open concurrently: the OpenAI/Fireworks SSE stream interleaves deltas across tool-call indices. Map keyed by ToolCallId.
- At most one **thought** open at a time: `reasoning_content` is a single text channel per turn.
- At most one **message** open at a time: `content` is a single text channel per turn.
- The codec is responsible for closing the open thought/message before opening a tool call (the wire format dictates the ordering: `reasoning_content` → `content` → `tool_calls`).

#### 9.5.4 `getView()` / `transformMessage()`

`transformMessage` for `assistant_turn` produces an `LLMMessage` view. This is only consumed by **orphaned legacy code paths** (BAML compaction worker — stubbed in Phase 5; autopilot — orphaned; test-harness snapshots).

Since native code path doesn't use `getView` (the codec encodes Memory directly), `transformMessage` for `assistant_turn` produces a minimal text representation:

```ts
case 'assistant_turn': {
  const text = msg.parts.map(part => {
    switch (part._tag) {
      case 'thought':   return `<thought>${part.text}</thought>`
      case 'message':   return part.text
      case 'tool_call': return `<tool_call name="${part.toolName}">${JSON.stringify(part.input)}</tool_call>`
    }
  }).join('\n')
  return { role: 'assistant', content: textParts(text) }
}
```

This is *only* used by orphan paths; output format is a placeholder.

#### 9.5.5 `turn_outcome` no longer reads `CanonicalTurnProjection`

The handler now reads its own working state:

```ts
turn_outcome: ({ event, fork, getWorking }) => {
  const working = getWorking(event.forkId)
  const newMessages = [...fork.messages]
  if (working.parts.length > 0) {
    newMessages.push(new AssistantTurnMessage({
      turnId:     event.turnId,
      parts:      working.parts,
      strategyId: 'native',
    }))
  }
  // ... pending inbox results flushed as before ...
  return { ...fork, messages: newMessages, /* ... */ }
}
```

`CanonicalTurnProjection` is no longer read. Its source file stays but is unhooked.

### 9.6 `DisplayProjection` refactor

**File:** `packages/agent/src/projections/display.ts`

Display's `eventHandlers` is updated to consume the new event vocabulary. The output shape (`DisplayState`, `DisplayMessage`, `ThinkBlockMessage`, `AssistantMessageDisplay`, etc.) is **preserved** — CLI components don't change.

Mapping:

| Old handler | → New handler |
|---|---|
| `thinking_chunk` (text accumulate) | `thought_start` (create step), `thought_delta` (append), `thought_end` (close) |
| `thinking_end` | `thought_end` |
| `lens_start/chunk/end` | **deleted** (no lenses) |
| `message_start` | `assistant_message_start` |
| `message_chunk` | `assistant_message_delta` |
| `message_end` | `assistant_message_end` |
| `tool_event(ToolInputStarted)` | `tool_call_start` |
| `tool_event(ToolInputReady)` | `tool_call_end` |
| `tool_event(ToolExecutionStarted)` | `tool_execution_started` |
| `tool_event(ToolExecutionEnded)` | `tool_execution_ended` |
| `tool_event(ToolObservation)` | `tool_observation` |
| `tool_event(ToolParseError)` | (deleted — no parse errors on native; handled via `CodecDecodeError` if it surfaces) |

`raw_response_chunk` handler — deleted.

The `ThinkBlockStep` union retains shape: `{ type: 'thinking', content: string, isActive: bool }` and `{ type: 'tool', toolName, input?, state }`. Both still produced from the new events.

`ensureThinkBlock` logic, `streamingMessageId` tracking — same logic, different event names.

### 9.7 `TurnProjection` — minor

**File:** `packages/agent/src/projections/turn.ts`

Already mostly paradigm-agnostic. The internal `ToolCall[]` tracking switches from `tool_event` to `tool_call_start` / `tool_call_end` / `tool_execution_ended` events. Otherwise unchanged.

### 9.8 Orphaned projections

Unhooked from the projection registry — source files retained:
- `canonical-turn.ts`
- `canonical-xml.ts`
- `replay.ts`

Find the projection registry (in `packages/agent/src/index.ts` or similar — TBD via grep in implementation) and remove these from the active list. Their files compile or don't — we don't fix them.

### 9.9 Compaction projection

**File:** `packages/agent/src/projections/compaction.ts`

`estimateContentTokens` was called on `canonicalXml` from `CanonicalTurnProjection`. New source: estimate from `AssistantTurnMessage.parts` directly (no intermediate projection). Helper:

```ts
// packages/agent/src/projections/util/estimate-turn-tokens.ts
export const estimateAssistantTurnTokens = (msg: AssistantTurnMessage): number => {
  let tokens = 0
  for (const part of msg.parts) {
    switch (part._tag) {
      case 'thought':   tokens += estimateText(part.text); break
      case 'message':   tokens += estimateText(part.text); break
      case 'tool_call': tokens += estimateText(part.toolName) + estimateText(JSON.stringify(part.input)); break
    }
  }
  return tokens
}
```

`compaction.ts` reads from `MemoryProjection.fork.messages` (the latest assistant_turn message) instead of `CanonicalTurnProjection`. Single source of truth.

### 9.10 Tests

| Test | Change |
|---|---|
| `packages/agent/src/projections/__tests__/memory-queue-ordering.test.ts` | If it constructs `assistant_turn` literals with `content`, update to `parts: TurnPart[]`. |
| `packages/agent/src/projections/__tests__/canonical-turn.test.ts` | xml-act-only — orphaned. Disable via `.skip` (don't delete; lets future xml-act port revive). |
| `packages/agent/src/projections/__tests__/display-*.test.ts` | Update event-fixture inputs to use new event names. Output assertions (DisplayState shape) unchanged. |
| `packages/agent/src/inbox/__tests__/{render,render-results,compose}.test.ts` | Update to use new compose helpers (with `toolCallId`, no `query`). |
| `packages/agent/tests/memory/*.vitest.ts` | Use harness — should still work. Verify on first run. |
| New: `packages/agent/src/projections/__tests__/memory-native.test.ts` | Send a sequence of new-vocab events; assert `parts: TurnPart[]` accumulates correctly; assert `assistant_turn` message is appended on `turn_outcome`. |

### 9.11 Files modified / created in Phase 4

```
packages/agent/src/inbox/types.ts                   # re-export from codecs
packages/agent/src/inbox/compose.ts                 # add toolCallId param, remove query
packages/agent/src/inbox/render.ts                  # remove query references
packages/agent/src/inbox/render-results.ts          # remove query references
packages/agent/src/projections/memory.ts            # major refactor — new handlers
packages/agent/src/projections/display.ts           # event-name remap
packages/agent/src/projections/turn.ts              # minor: event names
packages/agent/src/projections/compaction.ts        # read from MemoryProjection
packages/agent/src/projections/util/estimate-turn-tokens.ts  # NEW
packages/agent/src/projections/__tests__/memory-native.test.ts  # NEW
```

Unhooked (in the active registry, files retained):
```
packages/agent/src/projections/canonical-turn.ts
packages/agent/src/projections/canonical-xml.ts
packages/agent/src/projections/replay.ts
```

### 9.12 Verification

- `bun typecheck` for `packages/agent` passes (xml-act-coupled orphan files may not — accepted).
- `memory-native.test.ts` passes.
- Inbox render tests pass.
- Display tests pass with updated event-name fixtures.

---

## 10. Phase 5 — TurnEngine + ToolDispatcher + Cortex rewrite

**Goal.** Build the new turn loop. `TurnEngine` and `ToolDispatcher` are clean Effect services. `Cortex` becomes a thin orchestrator. `TurnController` simplifies to implicit turn control (loop iff any tools were called).

### 10.1 `TurnEngine` Effect service

**Design contract.**

- `TurnEngine.runTurn(params)` produces a `Stream<TurnPartEvent>`. **No event publishing inside TurnEngine** — that's Cortex's job, because publishing is done via the `publish` parameter received by the worker handler (per `event-core`'s API).
- TurnEngine encapsulates: encode (codec) → send (driver) → decode (codec) — three orchestrated pure-ish steps. Errors from any step are mapped into `TurnEngineError` with the originating cause preserved.
- TurnEngine is a `Context.Tag` so it can be substituted in tests (e.g. a mock that emits a fixed event sequence).

**File:** `packages/agent/src/engine/turn-engine.ts` (new)

```ts
import { Context, Effect, Schema, Stream } from 'effect'
import * as HttpClient from '@effect/platform/HttpClient'
import { CodecEncodeError, CodecDecodeError, type EncodeOptions, type ToolDef } from '@magnitudedev/codecs'
import { type TurnPartEvent } from '@magnitudedev/codecs'
import { DriverError } from '@magnitudedev/drivers'
import type { BoundModel } from '../model/bound-model'
import type { Message } from '../projections/memory'

export interface TurnEngineRunParams {
  readonly model:   BoundModel
  readonly memory:  readonly Message[]
  readonly tools:   readonly ToolDef[]
  readonly options: EncodeOptions
}

/**
 * `cause` is stored as `unknown` because the union of three Schema.TaggedErrors
 * cannot be cleanly expressed as a Schema field type without re-decoding.
 * Callers branch on `cause._tag` explicitly.
 */
export class TurnEngineError extends Schema.TaggedError<TurnEngineError>()(
  'TurnEngineError',
  { cause: Schema.Unknown },
) {}

export type TurnEngineCause = CodecEncodeError | CodecDecodeError | DriverError

export interface TurnEngineShape {
  /**
   * Open the turn: encode the request, dispatch via driver, decode chunks → TurnPartEvents.
   * The returned Effect produces a Stream that, when drained, runs the turn end-to-end.
   *
   * Errors from any stage flow as `TurnEngineError` either at Effect-resolve time
   * (encode failures, driver-pre-stream failures) or as Stream errors (decode/transport).
   * The `cause` field stores the underlying `CodecEncodeError | CodecDecodeError | DriverError` instance.
   */
  readonly runTurn: (
    params: TurnEngineRunParams,
  ) => Effect.Effect<
    Stream.Stream<TurnPartEvent, TurnEngineError>,
    TurnEngineError,
    HttpClient.HttpClient
  >
}

export class TurnEngine extends Context.Tag('TurnEngine')<TurnEngine, TurnEngineShape>() {}
```

**Layer:**

```ts
import { Effect, Layer, Stream } from 'effect'

export const TurnEngineLive = Layer.succeed(TurnEngine, {
  runTurn: ({ model, memory, tools, options }) =>
    Effect.gen(function* () {
      // 1. Encode
      const wireRequest = yield* model.codec.encode(memory, tools, options).pipe(
        Effect.mapError(cause => new TurnEngineError({ cause })),
      )

      // 2. Send → Stream<WireChunk, DriverError>
      const chunkStream = yield* model.driver.send(wireRequest, {
        endpoint:  model.wireConfig.endpoint,
        authToken: extractToken(model.auth),
      }).pipe(
        Effect.mapError(cause => new TurnEngineError({ cause })),
      )

      // 3. Decode → Stream<TurnPartEvent, CodecDecodeError | DriverError>
      const eventStream = model.codec.decode(chunkStream).pipe(
        Stream.mapError(cause => new TurnEngineError({ cause })),
      )

      return eventStream
    }),
})

// AuthInfo discriminator helper (existing AuthInfo: { type:'api', key } | { type:'oauth', token, expiresAt })
const extractToken = (auth: AuthInfo): string =>
  auth.type === 'api' ? auth.key : auth.token
```

**Why return `Stream`, not a collected `RunResult`?** The caller (Cortex) needs to publish each `TurnPartEvent` as an AppEvent to the event bus *as it arrives* for streaming UX. If TurnEngine collected internally, Cortex would never see intermediate events. Cortex therefore:

1. Calls `TurnEngine.runTurn(...)` → gets a `Stream<TurnPartEvent>`.
2. Drains the stream: for each event, calls `publish(liftTurnPartEvent(event, forkId, turnId))`, plus folds into a local accumulator that captures `toolCalls`, `finishReason`, `usage`.
3. After drain, dispatches tool calls via `ToolDispatcher`.
4. Publishes `turn_outcome`.

The fold is small enough that it lives in Cortex (§10.3); TurnEngine stays minimal.

### 10.2 `ToolDispatcher` Effect service

**File:** `packages/agent/src/engine/tool-dispatcher.ts` (new)

```ts
import { Context, Effect, Either, Layer, Schema } from 'effect'
import type { PublishFn } from '@magnitudedev/event-core'
import type { CodingAgentEvent } from '../events'
import { ToolObservationResultItem, ToolErrorResultItem } from '@magnitudedev/codecs'
import { ToolRegistry, ToolNotFound } from './tool-registry'

export interface ToolCallRequest {
  readonly forkId:     string | null
  readonly turnId:     string
  readonly toolCallId: string
  readonly toolName:   string
  readonly input:      unknown
}

export interface ToolDispatcherShape {
  /**
   * Dispatch a single tool call.
   *
   * Side-effects (via `publish`):
   *   - ToolExecutionStarted at start
   *   - ToolExecutionEnded at end (regardless of outcome)
   *   - ToolObservation OR ToolError before returning
   *
   * Errors are caught internally and surfaced as ToolErrorResultItem. The Effect
   * channel never errors (`E = never`).
   *
   * `publish` is passed in by the caller (Cortex worker handler) — matching the
   * event-core pattern where workers receive `publish` as a parameter.
   */
  readonly dispatch: (
    call:    ToolCallRequest,
    publish: PublishFn<CodingAgentEvent>,
  ) => Effect.Effect<
    ToolObservationResultItem | ToolErrorResultItem,
    never,
    ToolRegistry
  >

  /**
   * Dispatch multiple tool calls concurrently. Returns results in input order.
   * Concurrency: unbounded (model emits parallel tool calls when independent).
   */
  readonly dispatchAll: (
    calls:   readonly ToolCallRequest[],
    publish: PublishFn<CodingAgentEvent>,
  ) => Effect.Effect<
    readonly (ToolObservationResultItem | ToolErrorResultItem)[],
    never,
    ToolRegistry
  >
}

export class ToolDispatcher extends Context.Tag('ToolDispatcher')<ToolDispatcher, ToolDispatcherShape>() {}
```

Live Layer:

```ts
export const ToolDispatcherLive = Layer.succeed(ToolDispatcher, {
  dispatch: ({ forkId, turnId, toolCallId, toolName, input }, publish) => Effect.gen(function* () {
    const registry = yield* ToolRegistry

    yield* publish({ type: 'tool_execution_started', forkId, turnId, toolCallId, toolName })

    const tool = yield* registry.lookup(toolName).pipe(
      Effect.catchTag('ToolNotFound', () => Effect.succeed(null)),
    )

    const startTime = Date.now()

    if (tool === null) {
      yield* publish({ type: 'tool_execution_ended', forkId, turnId, toolCallId, toolName, durationMs: 0 })
      const errorItem = new ToolErrorResultItem({
        toolCallId, tagName: toolName, status: 'error',
        message: `Unknown tool: ${toolName}`,
      })
      yield* publish({ type: 'tool_error', forkId, turnId, toolCallId, toolName, status: 'error', message: errorItem.message })
      return errorItem
    }

    // Decode input via tool's schema
    const decoded = yield* Schema.decodeUnknown(tool.inputSchema)(input).pipe(
      Effect.either,
    )

    if (Either.isLeft(decoded)) {
      const durationMs = Date.now() - startTime
      yield* publish({ type: 'tool_execution_ended', forkId, turnId, toolCallId, toolName, durationMs })
      const errorItem = new ToolErrorResultItem({
        toolCallId, tagName: toolName, status: 'error',
        message: `Input schema decode failed: ${formatParseError(decoded.left)}`,
      })
      yield* publish({ type: 'tool_error', forkId, turnId, toolCallId, toolName, status: 'error', message: errorItem.message })
      return errorItem
    }

    // Run the tool
    const outcome = yield* tool.execute(decoded.right).pipe(
      Effect.either,
    )
    const durationMs = Date.now() - startTime

    yield* publish({ type: 'tool_execution_ended', forkId, turnId, toolCallId, toolName, durationMs })

    if (Either.isRight(outcome)) {
      const item = new ToolObservationResultItem({
        toolCallId,
        tagName:    toolName,
        content:    outcome.right.content,    // ContentPart[]
      })
      yield* publish({ type: 'tool_observation', forkId, turnId, toolCallId, toolName, content: item.content })
      return item
    } else {
      const errorItem = new ToolErrorResultItem({
        toolCallId, tagName: toolName, status: 'error',
        message: outcome.left instanceof Error ? outcome.left.message : String(outcome.left),
      })
      yield* publish({ type: 'tool_error', forkId, turnId, toolCallId, toolName, status: 'error', message: errorItem.message })
      return errorItem
    }
  }),

  dispatchAll: (calls, publish) => Effect.forEach(calls, c => dispatch(c, publish), { concurrency: 'unbounded' }),
})
```

**Notes on the design.**

- `dispatch` and `dispatchAll` accept `publish: PublishFn<CodingAgentEvent>` as a parameter — matching the `event-core` pattern (workers receive `publish` as a callback to handlers). ToolDispatcher does NOT depend on a global `EventBus` Tag.
- **Parallelism.** `dispatchAll` runs calls with `concurrency: 'unbounded'`. The model emits parallel tool calls when it judges them independent. Within a turn, batch parallelism is safe by contract — sequencing across turns is the model's responsibility. If a tool needs serialization, it's the tool's responsibility to take a fork-scoped lock internally.
- **Per-turn ordering.** Results are returned in input order (`Effect.forEach` preserves order regardless of concurrency). The `MemoryProjection` folds them into `pendingInboxResults` in this order, so the resulting `InboxMessage.results` array matches the tool call order from the model's response.

`ToolRegistry` is a new Context.Tag that wraps the existing fork-scoped tool layer:

```ts
import { Schema } from 'effect'

export class ToolNotFound extends Schema.TaggedError<ToolNotFound>()(
  'ToolNotFound',
  { toolName: Schema.String },
) {}

export interface ToolRegistryShape {
  readonly lookup: (toolName: string) => Effect.Effect<RegisteredTool, ToolNotFound>
  readonly toolDefs: () => Effect.Effect<readonly ToolDef[]>
}
export class ToolRegistry extends Context.Tag('ToolRegistry')<ToolRegistry, ToolRegistryShape>() {}
```

The registry is constructed per-fork (via `Layer`) and provides typed access to the fork's `RegisteredTool`s. It replaces the implicit `RegisteredTool[]` array passed around in `ExecutionManager`.

**`makeToolRegistryLive(registeredTools): Layer<ToolRegistry>`** is the per-turn factory. The Cortex worker handler calls it on each `turn_started` to scope a fresh `ToolRegistry` to the fork's current toolset, then provides the layer to the `ToolDispatcher` calls inside the handler:

```ts
// inside Cortex turn_started handler
const toolSet = buildResolvedToolSet(agentDef, configState, modelSlot)
const toolRegistryLayer = makeToolRegistryLive(toolSet.tools)

yield* toolDispatcher.dispatchAll(calls, publish).pipe(
  Effect.provide(toolRegistryLayer),
)
```

`makeToolRegistryLive` impl:

```ts
export const makeToolRegistryLive = (tools: readonly RegisteredTool[]): Layer.Layer<ToolRegistry> =>
  Layer.succeed(ToolRegistry, {
    lookup: (toolName) => {
      const found = tools.find(t => t.name === toolName)
      return found
        ? Effect.succeed(found)
        : Effect.fail(new ToolNotFound({ toolName }))
    },
    toolDefs: () =>
      Effect.succeed(tools.map(t => ({
        name:        t.name,
        description: t.description,
        parameters:  JSONSchema.make(t.inputSchema),
      }))),
  })
```

### 10.3 `TurnController` simplification — implicit turn control

**File:** `packages/agent/src/workers/turn-controller.ts`

Current logic: reads `TurnProjection.fork.triggers[]` to decide whether to publish `turn_started`. Triggers come from `user_message`, `chain_continue` (set by `turn_outcome` based on `yieldTarget`), `task_created`, etc.

New logic: implicit turn control. After a turn ends, fire another turn iff any tool calls happened.

```ts
// After projections settle, for each fork:
const turn   = yield* read(TurnProjection, forkId)
const memory = yield* read(MemoryProjection, forkId)

const shouldStart =
  turn.isIdle &&
  !turn.isCompacting &&
  !turn.contextLimitBlocked &&
  hasPendingTrigger(turn, memory)

if (shouldStart) {
  yield* publish({
    type:    'turn_started',
    forkId,
    turnId:  newTurnId(),
    chainId: turn.activeChainId ?? newChainId(),
  })
}
```

(Events are plain TS interface objects, not class instances. Match `events.ts`.)

`hasPendingTrigger`:

```ts
const hasPendingTrigger = (turn: TurnState, memory: ForkMemoryState): boolean => {
  // Implicit turn control:
  // 1. Pending external trigger (user message, oneshot task, agent created, parent message, wake) — yes
  // 2. Last assistant turn called tools (and inbox now has results) — yes
  // 3. Otherwise — no

  if (turn.pendingExternalTriggers.length > 0) return true

  const lastMsg = memory.messages.at(-1)
  if (lastMsg?._tag === 'inbox' && lastMsg.results.some(isToolResult)) {
    // Tool results from the last turn are present → run another turn so model can act on them
    const lastAssistant = findLastAssistantTurn(memory.messages)
    if (lastAssistant && lastAssistant.parts.some(p => p._tag === 'tool_call')) return true
  }

  return false
}
```

`pendingExternalTriggers` replaces `TurnProjection.triggers` for external sources only (user message, oneshot, agent_created, parent_message, wake, interrupt). The `chain_continue` trigger is **removed** — turn continuation is implicit from tool calls.

Subagent yielding:
- A worker that wants to "yield back to its parent" simply doesn't call any tools on a turn. Its turn loop ends. Parent's turn controller fires when the subagent's `turn_outcome` event publishes a parent-targeted `parent_message` (via a `send_message_to_parent` tool the worker called *before* its no-tool turn).

Lead yielding to workers:
- A lead spawns workers via `spawn_worker` tool. The lead's turn ends naturally (no further tools, or just spawn_worker tool). The lead waits for `parent_message` events from workers (which become external triggers).

This is implicit turn control. No explicit yield targets.

### 10.4 `TurnOutcome` — kept rich, augmented

The existing `TurnOutcome` union (`packages/agent/src/events.ts`) is rich and well-tuned for the FSM/recovery logic in `TurnController` and downstream consumers (`outcomeWillChainContinue`, `agent-status`, etc.):

```ts
type TurnOutcome =
  | { _tag: 'Completed';             completion: TurnCompletion }
  | { _tag: 'ParseFailure';          error: ParseFailureEvent }
  | { _tag: 'ProviderNotReady';      detail: ProviderNotReadyDetail }
  | { _tag: 'ConnectionFailure';     detail: ConnectionFailureDetail }
  | { _tag: 'ContextWindowExceeded' }
  | { _tag: 'OutputTruncated' }
  | { _tag: 'SafetyStop';            reason: SafetyStopReason }
  | { _tag: 'Cancelled';             reason: CancelledReason }
  | { _tag: 'UnexpectedError';       message: string; detail?: UnexpectedErrorDetail }
```

**Decision: keep this union. Modify only the `Completed` variant to remove `yieldTarget`** (implicit turn control eliminates the model-emitted yield).

```ts
// New:
type TurnCompletion =
  | { readonly _tag: 'Completed'
      readonly toolCallsCount:  number   // number of tool calls dispatched this turn
      readonly finishReason:    'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'
      readonly feedback:        readonly TurnFeedback[]
    }
```

`yieldTarget` is **removed**. `outcomeWillChainContinue` is rewritten:

```ts
export const outcomeWillChainContinue = (outcome: TurnOutcome): boolean => {
  switch (outcome._tag) {
    case 'Completed':              return outcome.toolCallsCount > 0
    case 'ParseFailure':           return false  // codec errors don't auto-retry; surface to user
    case 'ConnectionFailure':      return false  // ditto
    case 'ContextWindowExceeded':  return false  // requires compaction first
    case 'OutputTruncated':        return false  // ambiguous — surface to user
    default:                       return false
  }
}
```

**Mapping codec/driver errors → TurnOutcome variants.** In Cortex's error handler:

| `TurnEngineError.cause`            | TurnOutcome variant                                                                                   |
|------------------------------------|-------------------------------------------------------------------------------------------------------|
| `CodecEncodeError`                 | `UnexpectedError { message, detail: { _tag: 'EngineDefect' } }` — encode failure is a programming bug |
| `CodecDecodeError`                 | `ParseFailure { error: ... }` — model produced invalid wire output                                    |
| `DriverError { status: 401/403 }`  | `ProviderNotReady { detail: { _tag: 'AuthFailed', ... } }`                                             |
| `DriverError { status: 4xx/5xx }`  | `ConnectionFailure { detail: { _tag: 'ProviderError', httpStatus } }`                                  |
| `DriverError { status: null }`     | `ConnectionFailure { detail: { _tag: 'TransportError' } }`                                              |

`finishReason: 'length'` from the codec → `OutputTruncated`.
`finishReason: 'content_filter'` from the codec → `SafetyStop { reason: { _tag: 'Other', message: 'content_filter' } }`.

**Removed entirely:**
- `TurnYieldTarget` (`'user' | 'invoke' | 'worker' | 'parent'`)
- `TurnCompletion.yieldTarget`
- The `feedback: TurnFeedback[]` system stays — it's used to surface model-side problems (invalid message destination, oneshot retrigger). Native paradigm initially emits empty feedback; future tools (`send_message_to_*`) may emit `InvalidMessageDestination` feedback.

### 10.5 `Cortex` rewrite

**File:** `packages/agent/src/workers/cortex.ts`

`Cortex` is a `Worker.defineForked`. Event handlers receive `(event, publish, read)` per the event-core API. The `turn_started` handler now drives the entire turn end-to-end: it consumes the TurnEngine stream, publishes events as they arrive, accumulates tool calls, dispatches them, and publishes `turn_outcome`.

```ts
import { Effect, Stream } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { TurnEngine, TurnEngineError } from '../engine/turn-engine'
import { ToolDispatcher } from '../engine/tool-dispatcher'
import { ToolRegistry } from '../engine/tool-registry'
import { ModelResolver } from '@magnitudedev/providers'
import { liftTurnPartEvent } from './cortex/lift-event'
import { renderNativeSystemPrompt } from './cortex/native-system-prompt'
import { mapEngineErrorToOutcome } from './cortex/map-error-to-outcome'

interface TurnFold {
  toolCalls:    Map<string /* ToolCallId */, { toolName: string; input: unknown }>
  finishReason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'
  usage:        TurnUsageData | null
}

const initFold = (): TurnFold => ({
  toolCalls: new Map(),
  finishReason: 'other',
  usage: null,
})

const foldEvent = (acc: TurnFold, event: TurnPartEvent): TurnFold => {
  switch (event._tag) {
    case 'tool_call_start':
      acc.toolCalls.set(event.id, { toolName: event.toolName, input: undefined })
      return acc
    case 'tool_call_end': {
      const existing = acc.toolCalls.get(event.id)
      if (existing) acc.toolCalls.set(event.id, { ...existing, input: event.input })
      return acc
    }
    case 'turn_finish':
      acc.finishReason = event.reason
      return acc
    case 'turn_usage':
      acc.usage = {
        inputTokens:      event.inputTokens,
        outputTokens:     event.outputTokens,
        cacheReadTokens:  event.cacheReadTokens,
        cacheWriteTokens: event.cacheWriteTokens,
      }
      return acc
    default:
      return acc
  }
}

export const CortexWorker = Worker.defineForked<CodingAgentEvent>()({
  name: 'Cortex',
  forkLifecycle: { activateOn: 'agent_created', completeOn: 'agent_killed' },
  eventHandlers: {
    turn_started: (event, publish, read) => Effect.gen(function* () {
      const { forkId, turnId } = event

      const sessionContext = yield* read(SessionContextProjection)
      const agentStatus    = yield* read(AgentStatusProjection)  // non-forked projection — no forkId arg
      const memoryState    = yield* read(MemoryProjection, forkId)
      const turnEngine     = yield* TurnEngine
      const toolDispatcher = yield* ToolDispatcher
      const toolRegistry   = yield* ToolRegistry
      const modelResolver  = yield* ModelResolver

      const agentDef   = agentStatus.agentDef
      const boundModel = yield* modelResolver.resolve(agentDef.modelSlot).pipe(
        Effect.catchAll(err => Effect.gen(function* () {
          // ProviderNotReady etc.
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId: event.chainId,
            strategyId: 'native',
            outcome: mapResolverErrorToOutcome(err),
            inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null,
            providerId: null, modelId: null,
          })
          return null
        })),
      )
      if (boundModel === null) return

      const providerId = boundModel.model.providerId
      const modelId    = boundModel.model.id

      const systemMessage = renderNativeSystemPrompt(agentDef, sessionContext.skills ?? [])
      const fullMemory    = [systemMessage, ...memoryState.messages]
      const toolDefs      = yield* toolRegistry.toolDefs()

      // Open the turn (returns Stream<TurnPartEvent>)
      const eventStreamE = turnEngine.runTurn({
        model:   boundModel,
        memory:  fullMemory,
        tools:   toolDefs,
        options: { thinkingLevel: agentDef.thinkingLevel ?? 'medium' },
      })

      const eventStream = yield* eventStreamE.pipe(
        Effect.catchTag('TurnEngineError', (err: TurnEngineError) => Effect.gen(function* () {
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId: event.chainId,
            strategyId: 'native',
            outcome: mapEngineErrorToOutcome(err),
            inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null,
            providerId, modelId,
          })
          return null
        })),
      )
      if (eventStream === null) return

      // Drain stream: publish each event as AppEvent and fold into accumulator.
      const fold = yield* eventStream.pipe(
        Stream.runFoldEffect(initFold(), (acc, ev) => Effect.gen(function* () {
          yield* publish(liftTurnPartEvent(ev, forkId, turnId))
          return foldEvent(acc, ev)
        })),
        Effect.catchTag('TurnEngineError', (err) => Effect.gen(function* () {
          // Stream-time error (decode/transport)
          yield* publish({
            type: 'turn_outcome', forkId, turnId, chainId: event.chainId,
            strategyId: 'native',
            outcome: mapEngineErrorToOutcome(err),
            inputTokens: null, outputTokens: null, cacheReadTokens: null, cacheWriteTokens: null,
            providerId, modelId,
          })
          return null
        })),
      )
      if (fold === null) return

      // Dispatch tool calls. ToolDispatcher publishes ToolExecutionStarted/Ended/Observation/Error.
      const calls: ToolCallRequest[] = Array.from(fold.toolCalls.entries()).map(([id, v]) => ({
        forkId, turnId,
        toolCallId: id,
        toolName:   v.toolName,
        input:      v.input,
      }))
      if (calls.length > 0) {
        yield* toolDispatcher.dispatchAll(calls, publish)
      }

      // Publish turn_outcome
      yield* publish({
        type: 'turn_outcome', forkId, turnId, chainId: event.chainId,
        strategyId: 'native',
        outcome: {
          _tag: 'Completed',
          completion: {
            toolCallsCount: calls.length,
            finishReason:   fold.finishReason,
            feedback:       [],
          },
        },
        inputTokens:      fold.usage?.inputTokens ?? null,
        outputTokens:     fold.usage?.outputTokens ?? null,
        cacheReadTokens:  fold.usage?.cacheReadTokens ?? null,
        cacheWriteTokens: fold.usage?.cacheWriteTokens ?? null,
        providerId,
        modelId,
      })
    }),
  },
})
```

That's the entire turn handler. Compare to the current ~180-line cortex implementation. No paradigm branch, no protocol prompt, no grammar generation, no xml-act runtime.

**Key things to verify when implementing:**
- `Worker.defineForked` — confirm `forkLifecycle.activateOn` accepts `'agent_created'`. (This is what the existing cortex uses.)
- `read(MemoryProjection, forkId)` — confirm the `WorkerReadFn` overload for forked projections. (From `event-core/worker/defineForked.ts:128`: `read(projection, overrideForkId?)`.)
- `mapEngineErrorToOutcome(err)` — defined in §10.4. Returns the appropriate `TurnOutcome` variant.
- `mapResolverErrorToOutcome(err)` — handles ProviderNotReady cases from `ModelResolver`. Existing logic in cortex covers this.

### 10.6 `renderNativeSystemPrompt`

**File:** `packages/agent/src/workers/cortex/native-system-prompt.ts` (new)

```ts
export const renderNativeSystemPrompt = (
  agentDef:        MagnitudeRoleDef,
  skills:          readonly Skill[],
  sessionContext:  SessionContext,
): SessionContextMessage => {
  // agentDef.systemPrompt is a compiled template (see packages/agent/src/agents/*.ts —
  // each exports a definition with `systemPrompt: compilePromptTemplate(rawText)`).
  // The native variant uses prompt templates that exclude xml-act protocol sections.
  const roleText = agentDef.systemPrompt({
    sessionContext,
    skills,
    // ... per-template inputs ...
  })

  return new SessionContextMessage({
    content: textParts(roleText),
  })
}
```

Currently each agent definition compiles its prompt from a single `.txt` file (e.g. `packages/agent/src/agents/prompts/lead.txt`). These files contain xml-act-coupled sections:

- Keep: role description, traits, communication standards, work management semantics, available skills section.
- Remove: xml-act tag syntax, parameter format, yield syntax, response protocol, turn control rules, message routing tag syntax, magnitude interpreter mechanics.

The cleanest path: introduce native-variant prompt files at `packages/agent/src/agents/prompts/native/{lead,worker,oneshot,planner,builder,debugger}.txt` and have agent definitions select between xml-act and native variants based on a `paradigm` flag — OR, since xml-act is being orphaned in this plan, just **rewrite the existing prompt files** as native-only (the xml-act variant lives in git history). The agent definitions stay structurally unchanged; the prompt template content is what changes.

**Decision: rewrite the existing prompt files** rather than fork. Simpler. xml-act revival, if ever undertaken, can pull from git history.

### 10.7 `extract-tool-defs` and `ToolRegistry` Layer construction

**File:** `packages/agent/src/engine/tool-registry-live.ts` (new)

```ts
export const makeToolRegistryLive = (registeredTools: readonly RegisteredTool[]): Layer.Layer<ToolRegistry> =>
  Layer.succeed(ToolRegistry, {
    lookup: (toolName) =>
      Effect.fromNullable(registeredTools.find(t => t.name === toolName)).pipe(
        Effect.mapError(() => new ToolNotFound({ toolName })),
      ),
    toolDefs: () => Effect.succeed(registeredTools.map(rt => new ToolDef({
      name:        rt.name,
      description: rt.description,
      parameters:  jsonSchemaFromEffectSchema(rt.inputSchema),
    }))),
  })
```

`jsonSchemaFromEffectSchema` is a helper that converts an Effect `Schema.Class` to a JSON Schema object. Effect exports `JSONSchema` as a top-level module — `JSONSchema.make(schema)` produces the JSON Schema. Import:

```ts
import * as JSONSchema from 'effect/JSONSchema'
// then:
const jsonSchema = JSONSchema.make(tool.inputSchema)
```

(`Schema.JSONSchema` is **not** a valid path. Earlier draft had it wrong.)

### 10.8 Removing `ExecutionManager.execute(xmlStream)`

**File:** `packages/agent/src/execution/execution-manager.ts`

The `execute()` method is no longer called from the live path (cortex bypasses it). Source file stays. `ExecutionManager`'s remaining responsibilities (fork init/teardown, observables, registered tools layer building) are extracted into:

- `ForkLifecycle` Effect service (`Context.Tag`) — `initFork`, `disposeFork`, `fork` (no change in semantics; just renamed and given a clean Tag).
- `ObservableSource` Effect service (`Context.Tag`) — `getObservables(forkId)`.

`ExecutionManager.execute()` itself is removed from the file (or stubbed to throw `not_implemented`). The xml-act `TurnEngine` import is removed.

`buildRegisteredTools(...)` and tool-layer construction logic move to a new file `packages/agent/src/engine/build-tool-layer.ts` and are consumed by `ToolRegistryLive`.

### 10.9 New tools — `send_message_to_user`, `send_message_to_parent`

**Phase 5 introduces two built-in tools** that workers and lead agents use to communicate without yield syntax. Per R6, with implicit turn control there is no `<yield_user/>` or `<yield_worker/>` — agents must use these tools to surface messages.

**File:** `packages/agent/src/tools/send-message.ts` (NEW). Pattern matches `packages/agent/src/tools/agent-tools.ts`:

```ts
import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { Fork, WorkerBusTag } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

const { ForkContext } = Fork

const SendMessageError = ToolErrorSchema('SendMessageError', {})

export const sendMessageToUser = defineTool({
  name:        'send_message_to_user' as const,
  group:       'communication' as const,
  description: 'Send a message to the human user. Use to ask questions, share progress, or hand off the turn to the user.',
  inputSchema: Schema.Struct({
    message: Schema.String.annotations({ description: 'Message text to send to the user' }),
  }),
  outputSchema: Schema.Struct({ delivered: Schema.Boolean }),
  errorSchema: SendMessageError,
  execute: (input, _ctx) => Effect.gen(function* () {
    const { forkId } = yield* ForkContext
    const bus = yield* WorkerBusTag<AppEvent>()
    yield* bus.publish({
      type:      'agent_message_to_user',
      forkId,
      message:   input.message,
      timestamp: Date.now(),
    })
    return { delivered: true }
  }),
  label: () => 'Sending message to user',
})

export const sendMessageToParent = defineTool({
  name:        'send_message_to_parent' as const,
  group:       'communication' as const,
  description: 'Send a message to the parent agent. Workers use this to communicate progress, ask questions, or hand off back to the lead.',
  inputSchema: Schema.Struct({
    message: Schema.String.annotations({ description: 'Message text to send to parent' }),
  }),
  outputSchema: Schema.Struct({ delivered: Schema.Boolean }),
  errorSchema: SendMessageError,
  execute: (input, _ctx) => Effect.gen(function* () {
    const { forkId } = yield* ForkContext
    if (forkId === null) {
      return yield* Effect.fail({
        _tag: 'SendMessageError' as const,
        message: 'Root agent has no parent to message.',
      })
    }
    const bus = yield* WorkerBusTag<AppEvent>()
    yield* bus.publish({
      type:      'parent_message',
      forkId,                  // sender fork
      message:   input.message,
      timestamp: Date.now(),
    })
    return { delivered: true }
  }),
  label: () => 'Sending message to parent',
})
```

(Verify `WorkerBusTag<AppEvent>()` form matches the existing usage of the worker bus from a tool execution context. If the actual existing tool publishes events differently — e.g. via an `EventEmitter` injected via `_ctx` — adapt accordingly. The existing `agent-tools.ts` uses `ExecutionManager.fork()` rather than direct event publishing; for the message tools we want event publishing — confirm by reading `WorkerBusTag` import path during implementation.)

**Events.** `agent_message_to_user` and `parent_message` are AppEvent variants. `parent_message` already exists in the codebase (used by current xml-act `<message dst="parent">`); `agent_message_to_user` is new — add to `events.ts` (interface form, like other AppEvents). The CLI's display projection already consumes a similar event for assistant text; reuse the renderer.

**Tool registration.** Both tools are added to the default tool list for all native agents. `packages/agent/src/agents/registry.ts`'s `defaultTools` list (or equivalent) — add `sendMessageToUser`, `sendMessageToParent`. The lead agent gets only `sendMessageToUser`; workers get both. Verified by reading the existing tool-registry to confirm the right registration site.

(If an agent emits a turn that calls neither a tool nor `send_message_to_*`, the implicit turn control halts that fork. Workers ending without a `send_message_to_parent` call simply go idle silently — that's a valid "I'm done" signal, but doesn't notify the parent. Document in the worker's system prompt that they should call `send_message_to_parent` before going idle if they want the parent to act.)

### 10.10 Files modified / created in Phase 5

```
packages/agent/src/engine/turn-engine.ts                # NEW — TurnEngine Tag + Live Layer
packages/agent/src/engine/tool-dispatcher.ts            # NEW — ToolDispatcher Tag + Live Layer
packages/agent/src/engine/tool-registry.ts              # NEW — ToolRegistry Tag + ToolNotFound + makeToolRegistryLive
packages/agent/src/engine/build-tool-layer.ts           # NEW — extracted from ExecutionManager
packages/agent/src/engine/fork-lifecycle.ts             # NEW — extracted lifecycle service
packages/agent/src/engine/observable-source.ts          # NEW — extracted observables
packages/agent/src/engine/index.ts                      # NEW — exports

packages/agent/src/tools/send-message.ts                # NEW — sendMessageToUser, sendMessageToParent

packages/agent/src/workers/cortex.ts                    # major rewrite — thin orchestrator
packages/agent/src/workers/cortex/native-system-prompt.ts # NEW
packages/agent/src/workers/cortex/lift-event.ts         # NEW (Phase 3, listed for Phase 5 file accounting)
packages/agent/src/agents/prompts/native/lead.txt       # NEW
packages/agent/src/agents/prompts/native/worker.txt     # NEW
packages/agent/src/agents/prompts/native/oneshot.txt    # NEW

packages/agent/src/workers/turn-controller.ts           # simplified — implicit turn control
packages/agent/src/projections/turn.ts                  # TurnOutcome shape simplified
packages/agent/src/events.ts                            # TurnOutcomeEvent shape updated; agent_message_to_user added

packages/agent/src/execution/execution-manager.ts       # execute() removed/stubbed; lifecycle extracted

packages/agent/src/agents/registry.ts                   # native agent variant tool-list registration
```

Orphaned (kept on disk, unhooked):
- `packages/agent/src/workers/cortex.ts`'s xml-act-specific helpers (`renderSystemPrompt`, `buildAckTurns`, `generateToolGrammar` callsites — removed from cortex; helper files stay)
- `packages/xml-act/`

### 10.10 Tests

| Test | Coverage |
|---|---|
| `engine/__tests__/turn-engine.test.ts` | Mock codec + driver. Run a turn. Assert AppEvents are published in order. Assert `TurnEngineRunResult` is correct. |
| `engine/__tests__/tool-dispatcher.test.ts` | Mock ToolRegistry with one tool. Dispatch a call. Assert events published, return value correct. Test schema decode failure path. Test tool-not-found path. Test tool-execution-error path. |
| `workers/__tests__/turn-controller-implicit.test.ts` | Last assistant turn had tool calls + inbox has results → `turn_started` fires. Last assistant turn had no tool calls → no `turn_started`. External trigger queued → `turn_started` fires. |
| `workers/__tests__/cortex-native.test.ts` | Mock TurnEngine, ToolDispatcher, ModelResolver. Drive a `turn_started` event. Assert: `runTurn` called, `dispatchAll` called with correct calls, `turn_outcome` published. |

### 10.11 Verification

- All Phase 5 unit tests pass.
- `bun typecheck` for `packages/agent` passes (orphaned files may not — accepted).
- Existing turn-lifecycle tests are updated to new `TurnOutcome` shape and pass.

---

## 11. Phase 6 — Provider wiring & end-to-end

**Goal.** Compose `BoundModel` with a Driver + Codec. Wire up a Kimi K2.6 / Fireworks model record. Run end-to-end.

### 11.1 `BoundModel` refactor

**File:** `packages/providers/src/model/bound-model.ts`

```ts
import type { Codec } from '@magnitudedev/codecs'
import type { Driver } from '@magnitudedev/drivers'

export interface BoundModel {
  readonly model:         ProviderModel
  readonly canonicalModel: CanonicalModel | null
  readonly auth:          AuthInfo
  readonly driver:        Driver<unknown, unknown>
  readonly codec:         Codec<unknown, unknown>
  readonly wireConfig: {
    readonly endpoint:        string
    readonly wireModelName:   string
    readonly defaultMaxTokens: number
  }
}
```

The legacy `invoke(CodingAgentChat, ...)` method is **removed**. There is no longer any consumer for it on the live path.

### 11.2 `ModelConnection` simplification

**File:** `packages/providers/src/model/model-connection.ts`

The tagged enum is **deleted**. Connection data is just `AuthInfo` plus a base URL — both already on `BoundModel`.

### 11.3 Driver/Codec bindings registry

**File:** `packages/providers/src/model/protocol-bindings.ts` (new)

```ts
export interface CodecConfig {
  readonly wireModelName:    string
  readonly defaultMaxTokens: number
  readonly supportsReasoning: boolean
  readonly supportsVision:    boolean
}

export interface ProtocolBindingsShape {
  readonly drivers: ReadonlyMap<string, Driver<unknown, unknown>>
  readonly codecs:  ReadonlyMap<string, (config: CodecConfig) => Codec<unknown, unknown>>
}
export class ProtocolBindings extends Context.Tag('ProtocolBindings')<ProtocolBindings, ProtocolBindingsShape>() {}
```

Live Layer:

```ts
export const ProtocolBindingsLive = Layer.succeed(ProtocolBindings, {
  drivers: new Map([
    ['openai-chat-completions', OpenAIChatCompletionsDriver],
  ]),
  codecs: new Map([
    ['native-chat-completions', NativeChatCompletionsCodec],
  ]),
})
```

### 11.4 `ProviderDefinition` updates

**File:** `packages/providers/src/types.ts`, `packages/providers/src/registry.ts`

```ts
interface ProviderDefinition {
  readonly id:                string
  readonly name:              string
  readonly defaultBaseUrl:    string
  readonly authMethods:       readonly AuthMethod[]
  readonly inventoryMode:     'static' | 'dynamic'
  readonly providerFamily?:   'cloud' | 'local'

  // NEW — protocol bindings:
  readonly driverId:          string
  readonly codecId:           string

  readonly models:            readonly ProviderModel[]
}
```

`resolveProtocol(auth)` is removed.

### 11.5 New Fireworks (native) provider entry

```ts
{
  id: 'fireworks-native',
  name: 'Fireworks AI',
  defaultBaseUrl: 'https://api.fireworks.ai/inference/v1',
  authMethods: [{ type: 'api-key', envKeys: ['FIREWORKS_API_KEY'] }],
  inventoryMode: 'static',
  providerFamily: 'cloud',
  driverId: 'openai-chat-completions',
  codecId:  'native-chat-completions',
  models: [
    {
      id: 'accounts/fireworks/models/kimi-k2p6',
      providerId: 'fireworks-native',
      providerName: 'Fireworks AI',
      modelId: ModelId.makeUnsafe('kimi-k2.6'),
      name: 'Kimi K2.6',
      contextWindow: 256_000,
      maxOutputTokens: 16_384,
      supportsToolCalls: true,
      supportsReasoning: true,
      supportsVision:    true,
      supportsGrammar:   false,
      costs: null,
    },
  ],
}
```

### 11.6 `ModelResolver.resolve` refactor

**File:** `packages/providers/src/resolver/model-resolver-live.ts`

```ts
import { Context, Effect, Layer } from 'effect'
import { ModelResolver, type ModelResolverShape } from './model-resolver'
import { ProviderState } from '../runtime/contracts'
import { ProviderAuth, AppConfig, ProviderCatalog } from '../runtime/contracts'
import { ProtocolBindings } from '../model/protocol-bindings'
import { ensureAuth } from './ensure-auth'
import { NotConfigured, AuthMissing, ProviderNotFound, BindingMissing } from '../errors'
import { getProvider } from '../registry'
import { getModel as getCanonicalModel } from '../model/generated'
import type { BoundModel, ModelSlot } from '../model/bound-model'

export const makeModelResolver = (): Layer.Layer<
  ModelResolver,
  never,
  ProtocolBindings | ProviderState | ProviderAuth | ProviderCatalog | AppConfig
> =>
  Layer.effect(
    ModelResolver,
    Effect.gen(function* () {
      const bindings = yield* ProtocolBindings
      const state    = yield* ProviderState
      const shape: ModelResolverShape<string> = {
        resolve: (slot) =>
          Effect.gen(function* () {
            const peek = yield* state.peek(slot)
            if (!peek) {
              return yield* Effect.fail(
                new NotConfigured({ message: `No model configured for slot: ${slot}` }),
              )
            }
            const { model: providerModel, auth } = peek
            const providerDef = getProvider(providerModel.providerId)
            if (!providerDef) {
              return yield* Effect.fail(
                new ProviderNotFound({ providerId: providerModel.providerId }),
              )
            }
            if (!auth) {
              return yield* Effect.fail(new AuthMissing({ slot }))
            }
            // ensureAuth needs ProviderState | ProviderAuth in its R; we already have both via Layer.
            yield* ensureAuth(slot)

            const driver = bindings.drivers.get(providerDef.driverId)
            if (!driver) {
              return yield* Effect.fail(
                new BindingMissing({ kind: 'driver', id: providerDef.driverId }),
              )
            }
            const codecFactory = bindings.codecs.get(providerDef.codecId)
            if (!codecFactory) {
              return yield* Effect.fail(
                new BindingMissing({ kind: 'codec', id: providerDef.codecId }),
              )
            }

            const codec = codecFactory({
              wireModelName:     providerModel.id,
              defaultMaxTokens:  providerModel.maxOutputTokens ?? 4096,
              supportsReasoning: providerModel.supportsReasoning,
              supportsVision:    providerModel.supportsVision,
            })

            const boundModel: BoundModel = {
              model: providerModel,
              canonicalModel:
                providerModel.modelId !== null
                  ? getCanonicalModel(providerModel.modelId)
                  : null,
              auth,
              driver,
              codec,
              wireConfig: {
                endpoint:         providerDef.defaultBaseUrl,
                wireModelName:    providerModel.id,
                defaultMaxTokens: providerModel.maxOutputTokens ?? 4096,
              },
            }
            return boundModel
          }),
      }
      return shape
    }),
  )
```

Errors: `NotConfigured` (existing), `AuthMissing`, `ProviderNotFound`, `BindingMissing` — `Schema.TaggedError<X>()(...)` classes in `packages/providers/src/errors/`. The first three exist already; `ProviderNotFound` and `BindingMissing` are new in Phase 6.

**`ModelResolver` Tag style.** The existing tag is `Context.GenericTag<ModelResolverShape<string>>('ModelResolver')`. We **keep this style** (don't migrate to `Context.Tag`); the tag identity is preserved so existing consumers continue to work.

**On `BoundModel`'s `Driver<unknown, unknown>` / `Codec<unknown, unknown>` typing.** The driver's `WireRequest`/`WireChunk` types and codec's wire types must align — the codec produces what the driver accepts. We erase these types at the `BoundModel` boundary because:

1. The codec+driver pair is selected from a per-`ProviderDefinition` binding lookup at runtime; TypeScript can't statically prove that the runtime-looked-up driver and codec have matching wire types.
2. Outside the codec/driver internals, no consumer needs the wire types — only `TurnEngine.runTurn` (which closes over the `BoundModel`) exposes `Stream<TurnPartEvent>`.

The alignment invariant is enforced by **construction discipline**: each `ProviderDefinition.{driverId, codecId}` pair is documented to be wire-compatible, and binding-table entries in `ProtocolBindingsLive` only register pairs that share types. Mis-pairing is a programming error caught at integration test time, not at compile time. (A future refactor could introduce a typed `ProtocolPair<W, C>` registry that types-correlate driver+codec, but that's not needed for the native goal.)

### 11.7 Compaction worker stub

**File:** `packages/agent/src/workers/compaction-worker.ts`

Replace the body with a no-op that publishes a synthetic completion so the projection FSM doesn't get stuck. Native compaction is a separate effort.

### 11.8 Smoke test script

**Important.** `TestHarness` (in `packages/agent/src/test-harness/harness.ts`) **mocks the turn engine** via `MockTurnScript` — it does NOT exercise real driver/codec/provider stacks. It's for scripted-response unit-style tests. So the smoke test must use the real Magnitude runtime (the Live layers), not the harness.

**File:** `$M/scripts/native-e2e-smoke.ts`

```ts
import { Effect, Layer, Stream } from 'effect'
import { FetchHttpClient } from '@effect/platform'
import { Agent } from '@magnitudedev/event-core'
import { makeProviderRuntimeLive } from '@magnitudedev/providers'
import { ProtocolBindingsLive } from '@magnitudedev/providers/model/protocol-bindings'
import { TurnEngineLive } from '@magnitudedev/agent/engine/turn-engine'
import { ToolDispatcherLive } from '@magnitudedev/agent/engine/tool-dispatcher'
import { CortexWorker } from '@magnitudedev/agent/workers/cortex'
import { /* projections, other workers... */ } from '@magnitudedev/agent'

const LiveStack = Layer.mergeAll(
  FetchHttpClient.layer,      // satisfies HttpClient.HttpClient — required by OpenAIChatCompletionsDriver
  TurnEngineLive,
  ToolDispatcherLive,
  ProtocolBindingsLive,
  makeProviderRuntimeLive({ /* ... */ }),
  // ... all projections + workers ...
)

const program = Effect.gen(function* () {
  // Initialize the agent with the fireworks-native provider as the active model slot.
  const agent = yield* Agent.create({
    rootAgentVariant: 'lead-oneshot',
    initialModelSlot: { providerId: 'fireworks-native', modelId: 'accounts/fireworks/models/kimi-k2p6' },
  })

  // Send a user message
  yield* agent.publish({
    type: 'user_message',
    forkId: null,
    messageId: createId(),
    timestamp: Date.now(),
    content: textParts('List the files in /tmp using the shell tool, then tell me how many there are.'),
    attachments: [],
    mode: 'text',
    synthetic: false,
    taskMode: false,
  })

  // Wait for idle (no pending tool calls, no working turns)
  yield* agent.waitForIdle({ timeoutMs: 120_000 })

  // Inspect final memory
  const memory = yield* agent.read(MemoryProjection, null)
  for (const msg of memory.messages) {
    console.log('-', msg._tag, msg._tag === 'assistant_turn' ? `parts=${msg.parts.length}` : '')
  }
  const lastAssistant = memory.messages.findLast(m => m._tag === 'assistant_turn')
  if (lastAssistant) {
    console.log('Last assistant turn parts:')
    for (const part of lastAssistant.parts) {
      console.log(' ', part._tag, part._tag === 'thought' ? part.text.slice(0, 80) : part._tag === 'message' ? part.text.slice(0, 80) : `${part.toolName}(${JSON.stringify(part.input).slice(0, 80)})`)
    }
  }
})

await Effect.runPromise(program.pipe(Effect.provide(LiveStack)))
```

Run: `FIREWORKS_API_KEY=… bun run $M/scripts/native-e2e-smoke.ts`

**Important note on `Agent.create`** — the precise API for bootstrapping the live stack outside the CLI is in `packages/event-core/src/agent/define.ts` and `packages/agent/src/index.ts`. This script may need adjustment to match the actual constructor / layer composition. The CLI (`cli/src/app.tsx`) builds the same stack at startup; the smoke script can be modeled on that.

### 11.9 Live integration test

**File:** `packages/agent/tests/native-e2e.vitest.ts` — gated by `RUN_LIVE_TESTS=1`.

Composes the same live stack as the smoke script, drives a user message, asserts:
1. `turn_outcome` event publishes with `_tag: 'Completed'`.
2. Last assistant_turn message has at least one `MessagePart` (the model's final reply) or one `ToolCallPart` (if it tried to call a tool).
3. No `ParseFailure` / `ConnectionFailure` / `UnexpectedError` outcomes were emitted.

Skips with `it.skipIf(!process.env.RUN_LIVE_TESTS)`.

**NOT** using `TestHarness` (which mocks turns).

### 11.10 Files modified / created in Phase 6

```
packages/providers/src/model/bound-model.ts            # codec+driver fields, runTurn removed
packages/providers/src/model/model-connection.ts       # delete
packages/providers/src/model/model-driver.ts           # replace with bindings types
packages/providers/src/model/protocol-bindings.ts      # NEW — ProtocolBindings Tag + Layer
packages/providers/src/types.ts                        # ProviderDefinition gains driverId/codecId
packages/providers/src/registry.ts                     # add fireworks-native entry
packages/providers/src/resolver/model-resolver-live.ts # refactor: lookup bindings
packages/providers/src/runtime/live.ts                 # include ProtocolBindingsLive
packages/providers/src/errors/index.ts                 # ProviderNotFound, AuthMissing, BindingMissing

packages/agent/src/workers/compaction-worker.ts        # stubbed

$M/scripts/native-e2e-smoke.ts                         # NEW
packages/agent/tests/native-e2e.vitest.ts              # NEW
```

### 11.11 Verification

- `bunx --bun vitest run` passes for refactored agent code.
- `RUN_LIVE_TESTS=1 FIREWORKS_API_KEY=… bunx --bun vitest run packages/agent/tests/native-e2e.vitest.ts` passes.
- `FIREWORKS_API_KEY=… bun run $M/scripts/native-e2e-smoke.ts` runs end-to-end.
- CLI manual test: model selected as `fireworks-native:kimi-k2.6`; multi-turn conversation works; streaming thoughts/tool calls visible; idle on no-tool turn.

---

## 12. Risks, verification, decisions log

### 12.1 Risks & mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | Native API rejects custom `tool_call_id` format. | Empirically verified Fireworks/Kimi K2.6 accepts. Fallback (not implemented): switch to server-provided IDs with local↔wire mapping. |
| R2 | Some providers reject multimodal `role: 'tool'` content. | Empirically verified Fireworks/Kimi K2.6 accepts. Fallback (not implemented): codec moves images to follow-up user message. Configurable per-`Model`. |
| R3 | Streaming `delta.content` text appears alongside `delta.tool_calls` in tool-using flows. | Decoder closes any open thought/message before opening a tool call (and vice versa). Display handles both naturally. Memory persists thoughts, messages, **and** tool calls (per L12 — assistant text round-trips). |
| R4 | `tool_calls[].function.arguments` JSON is incomplete when finish_reason fires. | `tryParseJson` returns `{ _parseError: raw }`; the malformed input flows into `ToolCallEnd` (no separate error event). `ToolDispatcher`'s `Schema.decodeUnknown(tool.inputSchema)` rejects it; the result item is `ToolErrorResultItem { status: 'error', message: 'Input schema decode failed: ...' }`. The model sees this as the tool result on the next turn and can correct. |
| R5 | Implicit turn control loops forever. | Per-fork max-turns-per-chain limit (default 32). Existing `chainId` system preserves this. |
| R6 | Workers/parent communication broken without yield targets. | Workers use `send_message_to_parent` tool — defined and registered in §10.9 (`packages/agent/src/tools/send-message.ts`). The tool publishes a `parent_message` event; the parent's `TurnController` treats `parent_message` as an external trigger. Workers' turn loops end implicitly when they emit no tools on a turn (natural "I'm done" signal); workers should call `send_message_to_parent` before going idle if they want the parent to act. `send_message_to_user` (also new in §10.9) is the equivalent for lead→user communication. |
| R7 | Removed events break CLI components. | CLI consumes `DisplayState`. Display projection updated to consume new events but emits same `DisplayState` shape. Verified via display tests. |
| R8 | Compaction stubbed — long sessions overflow context. | Smoke test uses short conversations. Full compaction is a separate workstream. |
| R9 | `ToolCall` schema decode failure invisible to user. | `ToolDispatcher` emits `ToolErrorEvent`. Verify Display handler renders this. |
| R10 | Auth token refresh during a long stream. | Token snapshot at request time. If expires mid-stream, stream fails 401 → `TurnEngineError → TurnOutcome.DriverError`. Refreshed on next turn via existing `ensureAuth`. |
| R11 | `JSONSchema.make` of an Effect schema doesn't match what the model expects. | Effect's JSON Schema generator (`effect/JSONSchema`) handles common cases. Add per-tool optional `parametersJsonSchema?` override field; codec uses if present. (Not implemented; risk acknowledged.) |
| R12 | xml-act tests fail and block CI. | Skip via `.skip` or `vitest.config.ts` exclude patterns. |
| R13 | `bun typecheck` fails because orphan files reference removed types. | Exclude orphaned files from tsconfig via `exclude:` patterns. |

### 12.2 Verification matrix

| Layer | Verification | Phase |
|---|---|---|
| Driver SSE parser | `sse.test.ts` | 1 |
| Driver HTTP | `openai-chat-completions.test.ts` mocked + live | 1 |
| Codec encode | `encode.test.ts` per Message variant | 2 |
| Codec decode | `decode.test.ts` per chunk pattern | 2 |
| Codec end-to-end | `end-to-end.test.ts` against captured fixtures | 2 |
| Event vocabulary | `lift-event.test.ts` 1-to-1 | 3 |
| Memory projection | `memory-native.test.ts` accumulates parts | 4 |
| Display projection | existing `display-*.test.ts` updated | 4 |
| Inbox rendering | existing render tests with toolCallId | 4 |
| TurnEngine | `turn-engine.test.ts` mocked | 5 |
| ToolDispatcher | `tool-dispatcher.test.ts` mocked tool | 5 |
| TurnController | `turn-controller-implicit.test.ts` | 5 |
| Cortex orchestration | `cortex-native.test.ts` mocked engine | 5 |
| Provider resolver | `native-resolution.test.ts` | 6 |
| End-to-end agent | `native-e2e.vitest.ts` live + smoke script | 6 |

### 12.3 Resolved decisions log

| # | Decision | Rationale |
|---|---|---|
| L1 | Implicit turn control: loop iff tools were called. No explicit yield targets. | User directive. Eliminates `yieldTarget` enum + transition logic. |
| L2 | Message tag dropped. Assistant-to-user/parent communication is via tools (`send_message_to_user`, `send_message_to_parent`). | User directive. Uniform: all output is thought or tool call. |
| L3 | Lenses dropped. Replaced by `ThoughtPart.level: 'low'/'medium'/'high'`. | User directive. Spec §7.1. |
| L4 | Filter system dropped. Tools that need filtering accept the filter as part of their input. | User directive. Clean — uniform tool input shape. |
| L5 | xml-act paradigm not ported onto new contracts. xml-act source files orphaned, kept on disk. | User directive — not migrating now. |
| L6 | Compaction worker stubbed (no-op). Native compaction is a separate effort. | Out of scope for this plan; smoke tests stay short enough. |
| L7 | BAML driver, BAML function names, BAML client registry orphaned. | Same — not maintained on the live path. May not compile; excluded from tsconfig. |
| L8 | `ThoughtPart.level` fixed at `'medium'` initially. Per-agent config later. | Decoupled from immediate need. |
| L9 | `ToolCallId` format `call-{ord}-{ts36}`, generated codec-side, ignoring server IDs. | Empirically verified compatible. Single ID space. |
| L10 | Native system prompt is minimal — agent role text + active skill content. No protocol prompt. Tools declared via wire `tools: [...]` array. | Native paradigm doesn't need protocol prompts. |
| L11 | `Codec` and `Driver` are plain TS interfaces; instances are values held by `BoundModel`. `ProtocolBindings` is the only Context.Tag for codec/driver lookup. `TurnEngine`, `ToolDispatcher`, `ToolRegistry` are Context.Tag services. | Spec §11–13, §16. Composition over dispatch. |
| L12 | `MessageStart`/`Delta`/`End` events emitted by the codec for `delta.content` text are persisted as `MessagePart { id, text }` in `parts: TurnPart[]`. Encoded back to wire as `{ role: 'assistant', content: <text> }`. | Assistant text is part of the conversation — must round-trip on subsequent turns or the model loses sight of its own prior responses, breaking multi-turn coherence. (Reverses an earlier draft decision.) |
| L13 | `transformMessage()` for `assistant_turn` produces a placeholder text representation, only used by orphan paths (compaction stubbed per L6, autopilot orphaned per L7). Native code path doesn't call `getView()`. | The codec encodes Memory directly. `getView()` is legacy and slated for removal once orphans are cleaned up. |
| L14 | `ResultEntry`/`TimelineEntry` canonical definitions live in `packages/codecs`. `packages/agent/src/inbox/types.ts` re-exports. | Single source of truth. |

---

*End of plan.*
