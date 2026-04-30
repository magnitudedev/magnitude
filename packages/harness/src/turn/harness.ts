import { Effect, Stream, Layer, Ref, Queue } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  BoundModel,
  ConnectionError,
  StreamError,
  ToolDefinition,
  Prompt,
} from "@magnitudedev/ai"
import type { HarnessEvent } from "../events"
import type { HarnessHooks } from "../hooks"
import type { Toolkit, ToolkitRequirements } from "../tool/toolkit"
import type { HarnessToolErased } from "../tool/tool"
import { dispatch } from "./dispatcher"
import {
  CanonicalAccumulatorReducer,
  projectCanonical,
  EngineStateReducer,
  createToolHandleReducer,
  type CanonicalTurnState,
  type CanonicalAccumulator,
  type EngineState,
  type ToolHandleState,
} from "./reducers"

// ── Config ───────────────────────────────────────────────────────────

export interface HarnessConfig<
  TToolkit extends Toolkit<any> = Toolkit<any>,
  RHooks = never,
> {
  readonly model: BoundModel<any, ConnectionError, StreamError>
  readonly toolkit: TToolkit
  readonly hooks?: HarnessHooks<RHooks>
  readonly layer?: Layer.Layer<ToolkitRequirements<TToolkit> | RHooks>
  readonly initialState?: EngineState
}

// ── Harness ──────────────────────────────────────────────────────────

export interface Harness {
  /** Stream a model response, dispatch tool calls, and produce events.
   *  Returns a LiveTurn whose events stream is driven by the harness —
   *  the consumer reads events, reducers update refs automatically. */
  readonly runTurn: (
    prompt: Prompt,
  ) => Effect.Effect<
    LiveTurn,
    ConnectionError | StreamError,
    HttpClient.HttpClient
  >
  /** Create an empty turn for replaying a recorded event sequence.
   *  The consumer drives the turn by calling `feed` with each event. */
  readonly createReplayTurn: () => Effect.Effect<ReplayTurn>
  /** Tool definitions derived from the toolkit, for prompt assembly. */
  readonly getToolDefinitions: () => readonly ToolDefinition[]
}

/** A turn driven by the harness — events flow from the model stream
 *  through the dispatch pipeline. Consume `events` to observe progress;
 *  refs are updated automatically before each event is emitted. */
export interface LiveTurn {
  /** Stream of harness events, ending with TurnEnd. */
  readonly events: Stream.Stream<HarnessEvent>
  /** Canonical assistant message + tool results, updated after each event. */
  readonly canonicalTurn: Ref.Ref<CanonicalTurnState>
  /** Engine bookkeeping — tool call map, outcomes, stopped flag. */
  readonly engineState: Ref.Ref<EngineState>
  /** Per-tool-call state machines, driven by the toolkit's state models. */
  readonly toolHandles: Ref.Ref<ToolHandleState>
}

/** A turn driven by the consumer — call `feed` with recorded events
 *  to reconstruct state without running the model. Same reducers,
 *  same refs, same final state as a LiveTurn that saw the same events. */
export interface ReplayTurn {
  /** Feed a single event through all reducers and hooks. */
  readonly feed: (event: HarnessEvent) => Effect.Effect<void>
  /** Canonical assistant message + tool results, updated after each feed. */
  readonly canonicalTurn: Ref.Ref<CanonicalTurnState>
  /** Engine bookkeeping — tool call map, outcomes, stopped flag. */
  readonly engineState: Ref.Ref<EngineState>
  /** Per-tool-call state machines, driven by the toolkit's state models. */
  readonly toolHandles: Ref.Ref<ToolHandleState>
}

// ── createHarness ────────────────────────────────────────────────────

export function createHarness<
  TToolkit extends Toolkit<any>,
  RHooks = never,
>(config: HarnessConfig<TToolkit, RHooks>): Harness {
  const { toolkit, hooks, model } = config

  // Build tool definitions array from toolkit
  const toolDefs: ToolDefinition[] = []
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    const tool = entry.tool as HarnessToolErased
    toolDefs.push(tool.definition)
  }

  const toolHandleReducer = createToolHandleReducer(toolkit)

  // ── Shared ref creation ──────────────────────────────────────────

  function makeRefs() {
    return Effect.all({
      accRef: Ref.make(CanonicalAccumulatorReducer.initial),
      canonical: Ref.make(projectCanonical(CanonicalAccumulatorReducer.initial)),
      engine: Ref.make(config.initialState ?? EngineStateReducer.initial),
      handles: Ref.make(toolHandleReducer.initial),
    })
  }

  type Refs = Effect.Effect.Success<ReturnType<typeof makeRefs>>

  // ── Shared event feeding (reducers + optional hooks + optional queue) ──

  function makeFeedEvent(
    refs: Refs,
    eventQueue?: Queue.Queue<HarnessEvent>,
  ): (event: HarnessEvent) => Effect.Effect<void> {
    return (event: HarnessEvent): Effect.Effect<void> =>
      Effect.gen(function* () {
        // Step 1: Update all reducers
        const newAcc = yield* Ref.updateAndGet(refs.accRef, (s) =>
          CanonicalAccumulatorReducer.step(s, event),
        )
        yield* Ref.set(refs.canonical, projectCanonical(newAcc))
        yield* Ref.update(refs.engine, (s) => EngineStateReducer.step(s, event))
        yield* Ref.update(refs.handles, (s) => toolHandleReducer.step(s, event))

        // Step 2: onEvent hook — erased boundary, type coverage enforced by createHarness.
        if (hooks?.onEvent) {
          const onEventEffect = hooks.onEvent(event) as Effect.Effect<void, never, unknown>
          if (config.layer) {
            yield* (Effect.provide(onEventEffect, config.layer as Layer.Layer<unknown>) as Effect.Effect<void>)
          } else {
            yield* (onEventEffect as Effect.Effect<void>)
          }
        }

        // Step 3: Enqueue for stream consumers (live turns only)
        if (eventQueue) {
          yield* Queue.offer(eventQueue, event)
        }
      })
  }

  // ── createReplayTurn ─────────────────────────────────────────────

  function createReplayTurn(): Effect.Effect<ReplayTurn> {
    return Effect.gen(function* () {
      const refs = yield* makeRefs()
      const feed = makeFeedEvent(refs)
      return {
        feed,
        canonicalTurn: refs.canonical,
        engineState: refs.engine,
        toolHandles: refs.handles,
      }
    })
  }

  // ── runTurn ──────────────────────────────────────────────────────

  function runTurn(
    prompt: Prompt,
  ): Effect.Effect<
    LiveTurn,
    ConnectionError | StreamError,
    HttpClient.HttpClient
  > {
    return Effect.gen(function* () {
      // Get the model stream (may fail with ConnectionError)
      const modelStream = yield* model.stream(prompt, toolDefs)

      const refs = yield* makeRefs()
      const eventQueue = yield* Queue.unbounded<HarnessEvent>()
      const emitEvent = makeFeedEvent(refs, eventQueue)

      // Build dispatch — delegates all event processing and tool execution
      const processing = dispatch({
        modelStream,
        toolkit,
        hooks: hooks as HarnessHooks<unknown> | undefined,
        layer: config.layer as Layer.Layer<unknown> | undefined,
        initialEngineState: config.initialState,
        emit: emitEvent,
      })

      // Fork the dispatch processing and ensure queue shutdown on completion
      yield* Effect.fork(
        processing.pipe(
          Effect.ensuring(Queue.shutdown(eventQueue)),
        ),
      )

      // Build event stream from queue, ending at TurnEnd
      const eventStream: Stream.Stream<HarnessEvent> = Stream.fromQueue(eventQueue).pipe(
        Stream.takeUntil((event) => event._tag === "TurnEnd"),
      )

      return {
        events: eventStream,
        canonicalTurn: refs.canonical,
        engineState: refs.engine,
        toolHandles: refs.handles,
      }
    })
  }

  return {
    runTurn,
    createReplayTurn,
    getToolDefinitions: () => toolDefs,
  }
}
