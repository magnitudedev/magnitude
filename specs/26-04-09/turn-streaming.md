# Turn Streaming Pipeline Spec

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Key Components](#3-key-components)
4. [Event Vocabulary](#4-event-vocabulary)
5. [Completion Protocol](#5-completion-protocol)
6. [Lifecycle & Scope](#6-lifecycle--scope)
7. [Invariants](#7-invariants)
8. [Edge Cases](#8-edge-cases)
9. [Key Files](#9-key-files)
10. [Historical Context: Queue Truncation Bug](#10-historical-context-queue-truncation-bug)

---

## 1. Overview

The turn streaming pipeline carries structured events from model execution to the rest of the system. When a turn runs, the model's response is parsed into events (message chunks, tool calls, thinking deltas, etc.) that stream through a single ordered channel to the event bus, where projections, UI, and lifecycle management consume them.

The pipeline's core job: deliver every event — including the terminal `TurnResult` — reliably from producer to consumer, regardless of relative speed.

### Core Principles

- **Single ordered channel.** All events and completion signals travel through the same FIFO queue. There is no out-of-band completion mechanism.
- **Producers write events, not transport control.** Producers receive a narrow `TurnEventSink` interface. They cannot manipulate queue lifecycle, shutdown, or read state.
- **Completion is in-band.** Stream termination (success, failure, defect) is encoded as a terminal envelope in the same queue as payload events. This guarantees buffered events are consumed before the stream ends.
- **Consumers are decoupled from transport.** Downstream code sees `Stream<TurnEvent, ...>` and does not know about the internal envelope protocol.

---

## 2. Architecture

### Data Flow

```
Model response (streaming chunks from provider)
  ↓
Cortex / MockCortex (producer — orchestrates the turn)
  ↓ sink.emit(RawResponseChunk)
  ↓ delegates to
ExecutionManager.execute(xmlStream, options, sink)
  ↓ parses XML stream via xml-act runtime
  ↓ sink.emit(MessageStart | MessageChunk | MessageEnd | ToolEvent | ThinkingDelta | ...)
  ↓ returns ExecuteResult
  ↓
Cortex / MockCortex
  ↓ sink.emit(TurnResult)    ← terminal payload event
  ↓ producer returns
  ↓
createTurnStream internals
  ↓ enqueues Done envelope   ← terminal control signal
  ↓
Queue<Envelope> (internal, FIFO)
  ↓ repeated Queue.take
  ↓ Event envelope → emit TurnEvent
  ↓ Done envelope  → end stream
  ↓ Failure/Defect → fail stream
  ↓
Stream<TurnEvent, XmlRuntimeCrash | TurnError, Scope>
  ↓ consumed by
drainTurnEventStream(stream, forkId, turnId, publish)
  ↓ publishes each event to event bus
  ↓ extracts TurnResult
  ↓ returns { finalResult }
  ↓
Cortex / MockCortex
  ↓ publishes turn_completed with result + usage
  ↓
Event bus → TurnProjection, Display, UI, parent wake triggers
```

### Boundary Diagram

```
┌─────────────────────────────────────────────────────┐
│ Producer scope (forkScoped)                         │
│                                                     │
│  Cortex / MockCortex                                │
│    ├── model call → raw chunks                      │
│    ├── ExecutionManager.execute(sink)                │
│    │     └── xml-act parse → sink.emit(events...)   │
│    └── sink.emit(TurnResult)                        │
│                                                     │
└──────────────┬──────────────────────────────────────┘
               │ TurnEventSink.emit()
               ▼
┌──────────────────────────────────────────────────────┐
│ createTurnStream internals                           │
│                                                      │
│  Queue<Envelope>  ←  Event | Done | Failure | Defect │
│       │                                              │
│       ▼                                              │
│  Stream<TurnEvent>  (repeated take + interpret)      │
│                                                      │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│ Consumer scope                                       │
│                                                      │
│  drainTurnEventStream                                │
│    ├── publish events to bus                         │
│    └── extract TurnResult → return { finalResult }   │
│                                                      │
│  Cortex publishes turn_completed                     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### `createTurnStream`

**File:** `packages/agent/src/execution/types.ts`

Factory that wires the producer to a stream. Owns the internal envelope queue and completion protocol. Accepts a producer function that receives a `TurnEventSink`, returns a `Stream<TurnEvent, XmlRuntimeCrash | TurnError, R | Scope>`.

Responsibilities:
- Allocate internal `Queue<Envelope>`
- Construct `TurnEventSink` that wraps events into `Event` envelopes
- Fork the producer as a scoped fiber
- After producer completes, enqueue exactly one terminal envelope (`Done`, `Failure`, or `Defect`)
- Build the output stream by repeatedly taking from the envelope queue and interpreting

### `TurnEventSink`

**File:** `packages/agent/src/execution/types.ts`

```ts
export interface TurnEventSink {
  readonly emit: (event: TurnEvent) => Effect.Effect<void>
}
```

The write-only interface that producers use to emit events. This is the only producer-facing abstraction — producers cannot access the queue, shut it down, or inspect its state.

### `Cortex` / `MockCortex`

**Files:** `packages/agent/src/workers/cortex.ts`, `packages/agent/src/test-harness/mock-cortex.ts`

Turn producers. They orchestrate model calls, delegate XML parsing to `ExecutionManager`, and emit the terminal `TurnResult` after execution completes. After draining the stream, they publish `turn_completed` to the event bus.

`Cortex` is the production path. `MockCortex` is the test harness path — notably, `MockCortex` uses `catchAllCause` for error handling while `Cortex` uses `catchAll` (typed errors only).

### `ExecutionManager`

**File:** `packages/agent/src/execution/execution-manager.ts`

Receives the XML stream and a `TurnEventSink`. Runs the xml-act runtime, processes structured events (messages, tools, thinking, lenses), and emits each as a `TurnEvent` through the sink. Returns an `ExecuteResult` but does **not** emit `TurnResult` — that's the caller's responsibility.

### `drainTurnEventStream`

**File:** `packages/agent/src/workers/turn-event-drain.ts`

The stream consumer. Runs `Stream.runForEach` over the turn stream, publishing each non-terminal event to the event bus and capturing `TurnResult` when it arrives. After the stream ends, returns `{ finalResult }`. Dies if `TurnResult` was never observed.

---

## 4. Event Vocabulary

All events are tagged unions under `TurnEvent`:

| Event | Description |
|---|---|
| `RawResponseChunk` | Raw text chunk from the model provider |
| `MessageStart` | Begin a message to a destination |
| `MessageChunk` | Text fragment within a message |
| `MessageEnd` | End of a message |
| `ThinkingDelta` | Thinking/reasoning text chunk |
| `ThinkingEnd` | End of a thinking block |
| `LensStarted` | Start of a named thinking lens |
| `LensDelta` | Text within a lens |
| `LensEnded` | End of a named lens |
| `ToolEvent` | Tool lifecycle event (input ready, execution, result, emission) |
| `TurnResult` | **Terminal.** Carries `ExecuteResult` + `CallUsage`. Always last on success. |

---

## 5. Completion Protocol

### Internal Envelope Type

```ts
// File-local to createTurnStream, NOT exported
type Envelope =
  | { readonly _tag: 'Event'; readonly event: TurnEvent }
  | { readonly _tag: 'Done' }
  | { readonly _tag: 'Failure'; readonly error: XmlRuntimeCrash | TurnError }
  | { readonly _tag: 'Defect'; readonly cause: Cause.Cause<unknown> }
```

### Protocol

1. **During execution:** Producer calls `sink.emit(event)`. Each call enqueues `{ _tag: 'Event', event }` into the internal `Queue<Envelope>`.

2. **On producer success:** `createTurnStream` enqueues `{ _tag: 'Done' }` after the producer returns.

3. **On producer typed failure:** `createTurnStream` enqueues `{ _tag: 'Failure', error }` where `error` is the `TurnError` or `XmlRuntimeCrash`.

4. **On producer defect:** `createTurnStream` enqueues `{ _tag: 'Defect', cause }` preserving the original `Cause`.

5. **Stream interpretation:** The output stream repeatedly takes from the queue:
   - `Event` → emit the `TurnEvent`
   - `Done` → end the stream
   - `Failure` → fail the stream with the typed error
   - `Defect` → fail the stream with the original cause

### Why Not Queue.shutdown

`Stream.fromQueue` in Effect terminates when `Queue.shutdown` is called, but it does **not** guarantee draining buffered items first. If the producer is faster than the consumer, shutdown can cut ahead of buffered events in the queue, dropping them — including `TurnResult`. The in-band envelope protocol eliminates this race by making the completion signal travel through the same FIFO channel as events.

---

## 6. Lifecycle & Scope

### Producer Fiber

The producer is forked with `Effect.forkScoped`. This ties the producer fiber's lifetime to the stream's scope:

- When the stream scope is closed normally (after `Done` is consumed), the producer has already completed.
- When the stream scope is interrupted externally, `forkScoped` interrupts the producer fiber.

### Queue Shutdown

`Queue.shutdown` is **not** used for normal completion. The queue may be shut down as part of scope cleanup/interruption, but this is a resource cleanup concern, not a stream completion signal.

### Exactly-Once Terminal Envelope

The producer fork body uses `Effect.matchCauseEffect` to classify the producer's exit and enqueue exactly one terminal envelope. This prevents double-terminal or missing-terminal scenarios.

---

## 7. Invariants

1. **Terminal delivery.** If the producer successfully emits `TurnResult`, the consumer observes it before the stream ends.

2. **FIFO ordering.** Events are delivered in the order they were emitted. No reordering, no skipping.

3. **Exactly one terminal envelope.** Every stream lifecycle produces exactly one `Done`, `Failure`, or `Defect` envelope.

4. **`TurnResult` is last payload.** On success, `TurnResult` is the final `TurnEvent` before `Done`. No valid producer emits payload events after `TurnResult`.

5. **Failure propagation.** Producer typed failures reach the stream consumer as typed failures. Producer defects reach as defects. Neither is silently swallowed.

6. **Scope ownership.** The producer fiber is owned by the stream scope. External interruption of the scope interrupts the producer.

---

## 8. Edge Cases

### Fast Producer / Slow Consumer

The core correctness property. Because `Done` is enqueued after all `Event` envelopes and travels through the same FIFO queue, the consumer must observe every event (including `TurnResult`) before seeing `Done`. Backlog size is irrelevant to correctness.

### Producer Typed Failure

Events already enqueued before the failure remain in the queue. The `Failure` envelope is appended after them. The consumer observes partial delivery followed by a typed stream failure.

### Producer Defect

Same as typed failure, but the `Defect` envelope preserves the original `Cause`. The consumer observes partial delivery followed by a defect.

### Consumer Interruption

When the consumer scope is interrupted, `forkScoped` interrupts the producer fiber. The producer may enqueue a `Defect` envelope (with interruption cause), but the consumer may not observe it — this is acceptable because the consumer is already being torn down.

### Empty Turn

Producer emits zero events and returns successfully. `createTurnStream` enqueues `Done`. The stream ends immediately. This is only valid if the producer also emitted `TurnResult` (which `drainTurnEventStream` requires).

---

## 9. Key Files

| File | Role |
|---|---|
| `packages/agent/src/execution/types.ts` | `createTurnStream`, `TurnEventSink`, `TurnEvent`, `Envelope` (internal) |
| `packages/agent/src/workers/cortex.ts` | Production turn producer |
| `packages/agent/src/test-harness/mock-cortex.ts` | Test harness turn producer |
| `packages/agent/src/execution/execution-manager.ts` | XML execution, emits granular events through sink |
| `packages/agent/src/workers/turn-event-drain.ts` | Stream consumer, publishes to event bus, extracts `TurnResult` |
| `packages/agent/src/projections/turn.ts` | Turn lifecycle state machine (`idle` → `active` → `idle`) |
| `packages/agent/src/workers/turn-controller.ts` | Turn scheduling, depends on `TurnProjection` lifecycle |

---

## 10. Historical Context: Queue Truncation Bug

### The Bug (2026-04-09)

The original `createTurnStream` used `Queue.shutdown` as the stream completion signal. After the producer finished, a finalizer called `Queue.shutdown(queue)`. `Stream.fromQueue` in Effect terminates on shutdown by catching the interrupted take and converting it to end-of-stream — **without draining buffered items**.

Under backlog pressure (producer faster than consumer), this dropped queued events including `TurnResult`. Without `TurnResult`, `drainTurnEventStream` died with a defect. In production `Cortex`, this defect bypassed the typed `catchAll`, so neither `turn_completed` nor `turn_unexpected_error` was published. The turn remained permanently `active` in `TurnProjection`, blocking all further turns on that fork.

### The Fix

Replaced `Queue.shutdown` as EOF with the in-band envelope protocol described in this spec. Completion now travels through the same FIFO queue as events, making truncation impossible under normal operation.

### Attempted Mitigations That Did Not Work

- `Effect.yieldNow()` before shutdown: only gives the scheduler one opportunity to run the consumer. Insufficient with a large backlog.
- `Deferred` + `doneCheck` (`Stream.concat(queueStream, doneCheck)`): `doneCheck` only runs after `queueStream` ends, so it cannot rescue dropped items. It was an error-propagation tail, not a drain guarantee.
