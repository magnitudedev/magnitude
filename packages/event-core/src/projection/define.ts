/**
 * Projection.define() - State derived from events with signal support
 *
 * Uses eventHandlers/signalHandlers pattern.
 * ALL communication (events and signals) is SYNCHRONOUS.
 *
 * Supports cross-projection reads via the `reads` config option.
 */

import { Effect, SubscriptionRef, Context, Layer, PubSub } from 'effect'
import { ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
import { AmbientServiceTag } from '../core/ambient-service'
import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
import { type AmbientDef, type AmbientValueOf } from '../ambient/define'
import {
  Signal,
  fromDef,
  type SignalDef,
  type SignalEmitters,
  type AttachSourceState,
  type SignalValue,
  type SignalSourceState
} from '../signal/define'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProjectionInstance<TState> {
  readonly state: SubscriptionRef.SubscriptionRef<TState>
  readonly get: Effect.Effect<TState>
}

// Extract PubSub types from signals record (for Workers to subscribe async)
type SignalPubSubs<TSignals extends Record<string, Signal<unknown, unknown>>> = {
  [K in keyof TSignals]: TSignals[K] extends Signal<infer T, unknown> ? PubSub.PubSub<T> : never
}[keyof TSignals]

// ---------------------------------------------------------------------------
// Read Types
// ---------------------------------------------------------------------------

// Import forked types for state extraction
import type { ForkedProjectionInstance, ForkedProjectionResult, ForkedState } from './defineForked'

/**
 * Extract fork state type from a forked projection instance.
 * Returns S (the per-fork state), not ForkedState<S>.
 */
type ExtractForkState<I> =
  I extends ForkedProjectionInstance<infer S>
    ? S
    : never

/**
 * Extract state type for non-forked ReadFn.
 * Non-forked projections reading forked projections get ForkedState<S>.
 * Non-forked projections reading non-forked projections get S.
 */
type ExtractStateForNonForkedReader<I> =
  I extends ProjectionInstance<infer S>
    ? S
    : I extends ForkedProjectionInstance<infer S>
      ? ForkedState<S>
      : never

/**
 * Extract state type from any projection result for non-forked readers.
 */
export type StateOfProjection<P> =
  P extends { Tag: infer T }
    ? T extends Context.Tag<infer I, any>
      ? ExtractStateForNonForkedReader<I>
      : never
    : never

/**
 * Extract state type from any projection result for forked readers (same-fork access).
 */
export type StateOfProjectionForForkedReader<P> =
  P extends { Tag: infer T }
    ? T extends Context.Tag<infer I, any>
      ? I extends ProjectionInstance<infer S>
        ? S
        : I extends ForkedProjectionInstance<infer S>
          ? S
          : never
      : never
    : never

/** Union type for any projection result - uses any for event type to allow variance */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyProjectionResult =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ProjectionResult<string, any, any, any, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ForkedProjectionResult<string, any, any, any, any>

/** The read function type - constrained to declared dependencies */
export type ReadFn<TReads extends readonly AnyProjectionResult[]> =
  <P extends TReads[number]>(projection: P) => StateOfProjection<P>

export type AmbientReader<TAmbients extends readonly AmbientDef<any, any>[]> = {
  get<C extends TAmbients[number]>(def: C): AmbientValueOf<C>
}

// ---------------------------------------------------------------------------
// Signal Handler Builder Types
// ---------------------------------------------------------------------------

/**
 * Builder function type for creating signal handlers with full type inference.
 * The `on` function infers TSignal from the signal argument and uses it to type the handler.
 */
export type ProjectionSignalHandlerBuilder<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  <TSignal extends Signal<any, any>>(
    signal: TSignal,
    handler: (ctx: {
      value: SignalValue<TSignal> & { timestamp: number }
      source: SignalSourceState<TSignal>
      state: TState
      emit: SignalEmitters<TSignalDefs>
      read: ReadFn<TReads>
      ambient: AmbientReader<TAmbients>
    }) => TState
  ) => { signal: TSignal; handler: typeof handler }

export type ProjectionAmbientHandlerBuilder<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> =
  <C extends TAmbients[number]>(ambientDef: C, handler: (ctx: {
    value: AmbientValueOf<C>
    state: TState
    emit: SignalEmitters<TSignalDefs>
    read: ReadFn<TReads>
    ambient: AmbientReader<TAmbients>
  }) => TState) => { ambient: C; handler: typeof handler }

/**
 * Return type of signal handler builder - the { signal, handler } pair
 */
export type ProjectionSignalHandlerPair<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> =
  ReturnType<ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients>>

export type ProjectionAmbientHandlerPair<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> =
  ReturnType<ProjectionAmbientHandlerBuilder<TState, TSignalDefs, TReads, TAmbients>>

// ---------------------------------------------------------------------------
// Config and Result Types
// ---------------------------------------------------------------------------

export interface ProjectionConfig<
  TName extends string,
  TState,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> {
  readonly name: TName
  readonly initial: TState

  /** Signal definitions - will be transformed to Signal<T, TState> in the result */
  readonly signals?: TSignalDefs

  /**
   * Projections that this projection can read from.
   * Creates dependency edges in the execution graph.
   */
  readonly reads?: TReads

  /** Ambient dependencies that this projection can read from synchronously. */
  readonly ambients?: TAmbients

  /** Event handlers - pure reducers that return new state */
  readonly eventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      state: TState
      emit: SignalEmitters<TSignalDefs>
      read: ReadFn<TReads>
      ambient: AmbientReader<TAmbients>
    }) => TState
  }

  /**
   * Signal handlers - subscribe to signals from other projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients>
  ) => readonly ProjectionSignalHandlerPair<TState, TSignalDefs, TReads, TAmbients>[]

  /**
   * Ambient handlers - react to runtime ambient changes.
   * Use the builder pattern: `ambientHandlers: (on) => [on(ambientDef, handler)]`
   */
  readonly ambientHandlers?: (
    on: ProjectionAmbientHandlerBuilder<TState, TSignalDefs, TReads, TAmbients>
  ) => readonly ProjectionAmbientHandlerPair<TState, TSignalDefs, TReads, TAmbients>[]
}

/** Signal subscription info for tooling/visualization */
export interface SignalSubscription {
  /** Full signal name (e.g., "Fork/forkCreated") */
  readonly signal: string
  /** Source projection name (e.g., "Fork") */
  readonly source: string
}

export interface ProjectionResult<
  TName extends string,
  TState,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TAmbients extends readonly AmbientDef<any, any>[] = readonly []
> {
  /** Projection name - used for dependency graph */
  readonly name: TName

  /** Marker to identify standard (non-forked) projections */
  readonly isForked: false

  /** Names of projections this one reads from (for tooling/visualization) */
  readonly reads: readonly string[]

  /** Ambients this projection reads from (for tooling/visualization and typing) */
  readonly ambients: TAmbients

  /** Signals this projection subscribes to (for tooling/visualization) */
  readonly signalSubscriptions: readonly SignalSubscription[]

  /** Context tag for dependency injection */
  readonly Tag: Context.Tag<ProjectionInstance<TState>, ProjectionInstance<TState>>

  /** Layer that provides the projection instance and signal PubSubs */
  readonly Layer: Layer.Layer<
    ProjectionInstance<TState> | SignalPubSubs<AttachSourceState<TSignalDefs, TState>>,
    never,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ProjectionBusService<TEvent> | PubSub.PubSub<any>
  >

  /** Signals with source state attached - Signal<T, TState> */
  readonly signals: AttachSourceState<TSignalDefs, TState>
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/**
 * Define a Projection with event and signal handlers.
 *
 * @typeParam TEvent - The event union type
 * @typeParam TState - The projection's state type (explicit)
 *
 * @example
 * ```typescript
 * type MyState = { count: number }
 *
 * const MyProjection = Projection.define<MyEvent, MyState>()({
 *   name: 'My',
 *   initial: { count: 0 },
 *   signals: {
 *     changed: Signal.create<number>('My/changed')
 *   },
 *   reads: [OtherProjection],
 *   eventHandlers: {
 *     increment: ({ state, emit, read }) => {
 *       const other = read(OtherProjection)
 *       emit.changed(state.count + 1)
 *       return { count: state.count + 1 }
 *     }
 *   },
 *   signalHandlers: (on) => [
 *     on(OtherProjection.signals.someSignal, ({ value, source, state, emit, read }) => {
 *       // Full type inference on all parameters
 *       return state
 *     })
 *   ]
 * })
 *
 * // MyProjection.signals.changed is Signal<number, MyState>
 * ```
 */
export function define<TEvent extends BaseEvent, TState>() {
  return <
    TName extends string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
    TReads extends readonly AnyProjectionResult[] = readonly [],
    TAmbients extends readonly AmbientDef<any, any>[] = readonly []
  >(
    config: ProjectionConfig<TName, TState, TEvent, TSignalDefs, TReads, TAmbients>
  ): ProjectionResult<TName, TState, TEvent, TSignalDefs, TAmbients> => {
    const serviceName = `${config.name}Projection`
    const Tag = Context.GenericTag<ProjectionInstance<TState>>(serviceName)
    const BusTag = ProjectionBusTag<TEvent>()

    // Convert SignalDefs to Signals with source state attached
    const signalDefs = config.signals ?? {} as TSignalDefs
    const signals = {} as Record<string, Signal<unknown, TState>>
    for (const [key, def] of Object.entries(signalDefs)) {
      signals[key] = fromDef<unknown, TState>(def as SignalDef<unknown>, config.name)
    }
    const typedSignals = signals as AttachSourceState<TSignalDefs, TState>

    // Build PubSub layers for Workers to subscribe (async)
    type TSignals = AttachSourceState<TSignalDefs, TState>
    const signalEntries = Object.entries(signals)

    let SignalPubSubLayers: Layer.Layer<SignalPubSubs<TSignals>, never, never> =
      Layer.empty as Layer.Layer<SignalPubSubs<TSignals>, never, never>

    for (const [, signal] of signalEntries) {
      const signalLayer = Layer.scoped(signal.tag, PubSub.unbounded<unknown>())
      SignalPubSubLayers = Layer.merge(SignalPubSubLayers, signalLayer) as Layer.Layer<SignalPubSubs<TSignals>, never, never>
    }

    // Extract read dependency names
    const readDeps = (config.reads ?? []) as readonly AnyProjectionResult[]
    const ambients = (config.ambients ?? []) as TAmbients
    const allowedReadNames = new Set(readDeps.map(p => p.name))

    // Extract signal subscription metadata for tooling (signal names only, no handlers)
    const signalSubscriptions: SignalSubscription[] = []
    if (config.signalHandlers) {
      const extractSignalName: ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients> = (signal) => {
        signalSubscriptions.push({
          signal: signal.name,
          source: signal.name.split('/')[0]
        })
        return { signal, handler: () => config.initial }
      }
      config.signalHandlers(extractSignalName)
    }

    const LogicLayer = Layer.scoped(
      Tag,
      Effect.gen(function* () {
        const bus = yield* BusTag
        const ambientService = yield* AmbientServiceTag
        const stateRef = yield* SubscriptionRef.make(config.initial)

        for (const ambientDef of ambients) {
          yield* ambientService.register(ambientDef)
        }

        // Register read dependencies with the bus
        for (const dep of readDeps) {
          yield* bus.registerDependency(config.name, dep.name)
        }

        // Register state getter for this projection
        yield* bus.registerStateGetter(
          config.name,
          () => Effect.runSync(SubscriptionRef.get(stateRef)),
          false // not forked
        )

        // Build read function
        const makeReadFn = (): ReadFn<TReads> => {
          return <P extends TReads[number]>(projection: P): StateOfProjection<P> => {
            if (!allowedReadNames.has(projection.name)) {
              throw new Error(
                `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
              )
            }
            return bus.getProjectionState(projection.name) as StateOfProjection<P>
          }
        }

        const readFn = makeReadFn()
        const ambientReader: AmbientReader<TAmbients> = {
          get: <C extends TAmbients[number]>(def: C): AmbientValueOf<C> => ambientService.getValue(def)
        }

        // Get PubSubs for async worker subscriptions
        const pubsubs: Record<string, PubSub.PubSub<unknown>> = {}
        for (const [, signal] of signalEntries) {
          pubsubs[signal.name] = yield* signal.tag
        }

        // Pending signals - Effect-based to integrate with bus.queueSignal
        let pendingSignalEffects: Effect.Effect<void>[] = []

        // Build sync emit functions that queue signals to ProjectionBus
        // Also publishes to PubSub for workers (async subscribers)
        const emitters: Record<string, (value: unknown) => void> = {}
        for (const [key, signal] of signalEntries) {
          const pubsub = pubsubs[signal.name]
          emitters[key] = (value: unknown) => {
            // Queue effect to run after handler returns
            pendingSignalEffects.push(
              Effect.gen(function* () {
                const sourceState = yield* SubscriptionRef.get(stateRef)
                yield* bus.queueSignal(signal.name, value, sourceState)
                // Publish to PubSub for workers (async)
                yield* PubSub.publish(pubsub, value)
              })
            )
          }
        }

        const typedEmitters = emitters as SignalEmitters<TSignalDefs>

        // Flush pending signal queue effects (runs queued signals to bus)
        const flushPendingSignals = Effect.gen(function* () {
          const effects = pendingSignalEffects
          pendingSignalEffects = []
          for (const effect of effects) {
            yield* effect
          }
        })

        // Register signal handlers with the bus (SYNC dispatch during flush phase)
        if (config.signalHandlers) {
          const on: ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients> = (signal, handler) => ({
            signal,
            handler
          })
          const handlerPairs = config.signalHandlers(on)

          for (const { signal, handler } of handlerPairs) {
            yield* bus.registerSignalHandler(
              signal.name,
              (value, sourceState) => Effect.gen(function* () {
                yield* SubscriptionRef.update(stateRef, (currentState) =>
                  handler({
                    value,
                    source: sourceState,
                    state: currentState,
                    emit: typedEmitters,
                    read: readFn,
                    ambient: ambientReader
                  })
                )
                // Queue any signals emitted by this handler
                yield* flushPendingSignals
              }),
              serviceName
            )
          }
        }

        if (config.ambientHandlers) {
          const on: ProjectionAmbientHandlerBuilder<TState, TSignalDefs, TReads, TAmbients> = (ambientDef, handler) => ({
            ambient: ambientDef,
            handler
          })
          const handlerPairs = config.ambientHandlers(on)

          for (const { ambient, handler } of handlerPairs) {
            yield* bus.registerAmbientHandler(
              ambient.name,
              (value) => Effect.gen(function* () {
                yield* SubscriptionRef.update(stateRef, (currentState) =>
                  handler({
                    value: value as never,
                    state: currentState,
                    emit: typedEmitters,
                    read: readFn,
                    ambient: ambientReader
                  })
                )
                yield* flushPendingSignals
              }),
              serviceName
            )
          }
        }

        // Build combined event handler from individual handlers
        const eventHandler = (event: TEvent): Effect.Effect<void> => {
          if (!config.eventHandlers) return Effect.void

          const handler = config.eventHandlers[event.type as TEvent['type']]
          if (!handler) return Effect.void

          return Effect.gen(function* () {
            yield* SubscriptionRef.update(stateRef, (currentState) =>
              handler({
                event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                state: currentState,
                emit: typedEmitters,
                read: readFn,
                ambient: ambientReader
              })
            )
            // Queue any signals emitted by this handler
            yield* flushPendingSignals
          })
        }

        // Register with projection bus
        // Registration order follows layer build order, which respects signal dependencies
        const eventTypes = config.eventHandlers
          ? Object.keys(config.eventHandlers) as TEvent['type'][]
          : []

        yield* bus.register(eventHandler, eventTypes, serviceName)

        return {
          state: stateRef,
          get: SubscriptionRef.get(stateRef)
        }
      })
    )

    // Compose layers
    type LayerOutput = ProjectionInstance<TState> | SignalPubSubs<TSignals>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    type LayerInput = ProjectionBusService<TEvent> | PubSub.PubSub<any>

    const FullLayer = Layer.provideMerge(LogicLayer, SignalPubSubLayers) as Layer.Layer<
      LayerOutput,
      never,
      LayerInput
    >

    return {
      name: config.name,
      isForked: false as const,
      reads: readDeps.map(p => p.name),
      ambients,
      signalSubscriptions,
      Tag,
      Layer: FullLayer,
      signals: typedSignals
    }
  }
}
