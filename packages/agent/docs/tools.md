# Tool System Architecture

## Overview

The tool system provides a typed pipeline from raw tool events to display-ready state. It is structured around three core contracts with strict abstraction boundaries and a single dependency direction: **events flow in, typed state comes out.**

The tools package owns the contracts. Consumers (agent, CLI) provide implementations.

---

## Package Dependency Graph

```
┌─────────────────────┐
│   @magnitudedev/    │
│       tools         │  ← Contract package (no internal deps)
│                     │     Defines: ToolStateEvent, StateModel,
│                     │     ToolCallState, BaseState, Phase,
│                     │     StreamingInput, ToolResult
└────────┬────────────┘
         │
         │ depends on
         ▼
┌─────────────────────┐
│   @magnitudedev/    │
│      xml-act        │  ← Parser layer
│                     │     Defines: XML binding, SchemaAccumulator,
│                     │     streaming shape derivation
└────────┬────────────┘
         │
         │ depends on tools + xml-act
         ▼
┌─────────────────────┐
│   @magnitudedev/    │
│       agent         │  ← Runtime layer
│                     │     Provides: tool implementations, state models,
│                     │     event normalizer, model registry,
│                     │     DisplayProjection
└────────┬────────────┘
         │
         │ depends on agent + tools
         ▼
┌─────────────────────┐
│       cli           │  ← Presentation layer
│                     │     Provides: Display interface, per-tool
│                     │     display implementations, display registry
└─────────────────────┘
```

Dependencies flow strictly downward. The tools package has no knowledge of XML, agents, or UI. The agent package has no knowledge of rendering. Only the CLI knows how to render state.

---

## Contracts (tools package)

The tools package provides three core contracts and one generic implementation.

### 1. ToolStateEvent

The normalized event language. A discriminated union covering the full tool lifecycle:

- **Lifecycle:** `started`, `inputUpdated`, `inputReady`, `parseError`
- **Approval:** `awaitingApproval`, `approvalGranted`, `approvalRejected`
- **Execution:** `executionStarted`, `emission`, `completed`, `error`, `rejected`, `interrupted`

This is the common language between event producers and state models. Event producers (normalizers) translate raw events into this format. State models consume only this format.

Generic parameters: `ToolStateEvent<TInput, TOutput, TEmission, TStreaming>` — typed to each tool's specific input/output/emission/streaming types.

### 2. StateModel

A pure reducer contract. Defines how a tool's display state evolves over time.

```
initial: TState
reduce(state: TState, event: ToolStateEvent) → TState
```

Each tool defines a `StateModel`. The shell model knows that on `inputUpdated`, it should extract the command from `event.streaming.body`. The file-read model knows that on `completed`, it should count lines from the output.

All model states extend `BaseState` which includes `phase: Phase` (`streaming | executing | completed | error | rejected | interrupted`).

### 3. ToolCallState (generic implementation)

A stateful manager for a single tool call. Given a `StateModel`, it:

- Holds current `state: TState` (starts at `model.initial`)
- Holds current `streaming: TStreaming` (updated from `inputUpdated`/`inputReady` events)
- Provides `dispatch(event: ToolStateEvent)` — calls `model.reduce` internally
- Provides `snapshot()` — returns `{ state, streaming }` for storage

This saves consumers from manually calling `model.reduce` and tracking streaming updates. One method to feed events, one method to read state.

### Supporting Types

- **`BaseState` / `Phase`** — minimal base state all models extend
- **`StreamingInput`** — shape of accumulated streaming XML input (fields, body, children)
- **`ToolResult<TOutput>`** — execution outcome (success/error/rejected/interrupted with output/message)

---

## Abstraction Boundary

**The tools package boundary ends at state.** It provides the event language, the reducer contract, and a generic state manager. Everything downstream of state — rendering, summarizing, UI layout — is the consumer's domain.

This means:
- The tools package has no `Display` interface, no `render()`, no `summary()`
- Consumers define their own display system based on the typed state they receive
- Different consumers (TUI, web UI, test harness) can render the same state differently

---

## Agent Package (runtime layer)

### Event Normalizer

`ToolEventNormalizer` translates raw xml-act events into `ToolStateEvent`. It holds a `SchemaAccumulator` per call for streaming XML parsing. Pure translation — no state reduction, no model awareness.

```
Raw xml-act event → normalize() → ToolStateEvent
```

### Model Registry

Maps `toolKey → StateModel`. Provides `getModelForToolKey(toolKey)` which returns the appropriate model (or `defaultModel` for unknown tools).

### DisplayProjection

Orchestrates the pipeline. Maintains a `ToolCallState` per active tool call. On each raw tool event:

1. Normalizes via `ToolEventNormalizer` → `ToolStateEvent`
2. Creates a `ToolCallState` on `started` events (using the model from the registry)
3. Dispatches the normalized event via `callState.dispatch(event)`
4. Snapshots `callState.snapshot()` onto the tool step as `visualState: { state, streaming }`

The UI reads `step.visualState` to render.

---

## CLI Package (presentation layer)

The CLI owns its display system entirely:

- **`Display<TState>`** interface — `render(props) → ReactNode` + `summary(state) → string`
- **Per-tool display files** — each tool has a display that knows how to render its specific state type
- **Display registry** — maps `toolKey → Display`, with fallback to a default display

Think-block reads `step.visualState.state` from the projection, looks up the display by `step.toolKey`, and calls `display.render()`. Live-activity calls `display.summary()` for tab bar text.

---

## Event Flow

```
Raw xml-act events
    │
    ▼
ToolEventNormalizer (agent)
    │  translates to ToolStateEvent
    ▼
ToolCallState (tools)
    │  dispatches to StateModel.reduce
    ▼
Model State (agent)
    │  snapshotted onto projection step
    ▼
Display (cli)
    │  renders state to UI
    ▼
React elements
```

---

## Design Principles

1. **Contract boundary at state** — the tools package provides events→state; rendering is the consumer's problem
2. **Strict dependency direction** — tools → xml-act → agent → CLI, no cycles
3. **Compile-time type safety** — generic parameters thread through tool → binding → model, catching mismatches at compile time
4. **Runtime simplicity** — `ToolCallState.dispatch()` is one method; consumers don't manage reducers manually
5. **Consumer flexibility** — any consumer can define its own display system for the same typed state
6. **Separation of concerns** — parsing, state reduction, and rendering are independently evolvable
