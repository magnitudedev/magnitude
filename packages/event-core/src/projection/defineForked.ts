/**
 * Projection.defineForked() - Projection with per-fork state
 *
 * Syntactic sugar for projections that need state partitioned by forkId.
 * Does exactly two things:
 * 1. State is Map<forkId, ForkState> instead of just ForkState
 * 2. Handlers receive `fork` (specific fork's state) instead of the whole Map
 *
 * Events must have a `forkId: string | null` field.
 * null forkId = root agent.
 *
 * Supports cross-projection reads via the `reads` config option.
 * When reading another forked projection, automatically resolves to the same forkId.
 */

import { Effect, SubscriptionRef, Context, Layer, PubSub } from 'effect'
import { ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
import {
  Signal,
  fromDef,
  type SignalDef,
  type SignalEmitters,
  type AttachSourceState,
  type SignalValue,
  type SignalSourceState
} from '../signal/define'
import {
  type AnyProjectionResult,
  type StateOfProjection,
  type SignalSubscription
} from './define'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Event with forkId field (null = root agent)
 */
export interface ForkableEvent extends BaseEvent {
  forkId: string | null
}

/**
 * Internal state structure - Map of forkId to fork state
 */
export type ForkedState<TForkState> = {
  forks: Map<string | null, TForkState>
}

/**
 * Instance interface exposed to workers
 */
export interface ForkedProjectionInstance<TForkState> {
  /** Get state for a specific fork */
  readonly getFork: (forkId: string | null) => Effect.Effect<TForkState>
  /** Get all forks */
  readonly getAllForks: () => Effect.Effect<Map<string | null, TForkState>>
  /** Raw state ref (for subscriptions) */
  readonly state: SubscriptionRef.SubscriptionRef<ForkedState<TForkState>>
}

// Extract PubSub types from signals record
type SignalPubSubs<TSignals extends Record<string, Signal<unknown, unknown>>> = {
  [K in keyof TSignals]: TSignals[K] extends Signal<infer T, unknown> ? PubSub.PubSub<T> : never
}[keyof TSignals]

// ---------------------------------------------------------------------------
// Read Types for Forked Projections
// ---------------------------------------------------------------------------

/**
 * Read function for event handlers in forked projections.
 * Automatically resolves forked dependencies to the same forkId.
 */
export type ForkedReadFn<TReads extends readonly AnyProjectionResult[]> =
  <P extends TReads[number]>(projection: P) => StateOfProjection<P>

/**
 * Read function for signal handlers in forked projections.
 * For forked dependencies, returns the full ForkedState since there's no single forkId context.
 */
export type ForkedSignalReadFn<TReads extends readonly AnyProjectionResult[]> =
  <P extends TReads[number]>(projection: P) =>
    P extends ForkedProjectionResult<any, infer TForkState, any, any>
      ? ForkedState<TForkState>
      : StateOfProjection<P>

// ---------------------------------------------------------------------------
// Signal Handler Builder Types
// ---------------------------------------------------------------------------

/**
 * Builder function type for creating signal handlers with full type inference.
 * The `on` function infers TSignal from the signal argument and uses it to type the handler.
 */
export type ForkedSignalHandlerBuilder<
  TForkState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly []
> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  <TSignal extends Signal<any, any>>(
    signal: TSignal,
    handler: (ctx: {
      value: SignalValue<TSignal> & { timestamp: number }
      source: SignalSourceState<TSignal>
      state: ForkedState<TForkState>
      emit: SignalEmitters<TSignalDefs>
      read: ForkedSignalReadFn<TReads>
    }) => ForkedState<TForkState>
  ) => { signal: TSignal; handler: typeof handler }

/**
 * Return type of signal handler builder - the { signal, handler } pair
 */
export type ForkedSignalHandlerPair<
  TForkState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly []
> =
  ReturnType<ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads>>

// ---------------------------------------------------------------------------
// Config and Result Types
// ---------------------------------------------------------------------------

export interface ForkedProjectionConfig<
  TName extends string,
  TForkState,
  TEvent extends ForkableEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TReads extends readonly AnyProjectionResult[] = readonly []
> {
  readonly name: TName

  /** Initial state for each fork (used when a new forkId is first seen) */
  readonly initialFork: TForkState

  /** Signal definitions */
  readonly signals?: TSignalDefs

  /**
   * Projections that this projection can read from.
   * Creates dependency edges in the execution graph.
   */
  readonly reads?: TReads

  /**
   * Event handlers - receive fork state, return new fork state.
   * Framework extracts forkId from event, manages the Map.
   * 
   * Return null to delete the fork from the Map (useful for cleanup on fork_completed).
   */
  readonly eventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      fork: TForkState
      emit: SignalEmitters<TSignalDefs>
      read: ForkedReadFn<TReads>
    }) => TForkState | null
  }

  /**
   * Broadcast event handlers - called for every fork regardless of event.forkId.
   * Use for global events that should update all forks.
   */
  readonly broadcastEventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      forkId: string | null
      fork: TForkState
      emit: SignalEmitters<TSignalDefs>
      read: ForkedReadFn<TReads>
    }) => TForkState | null
  }

  /**
   * Signal handlers - subscribe to signals from other projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads>
  ) => readonly ForkedSignalHandlerPair<TForkState, TSignalDefs, TReads>[]
}

export interface ForkedProjectionResult<
  TName extends string,
  TForkState,
  TEvent extends ForkableEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>
> {
  readonly name: TName

  /** Marker to identify forked projections */
  readonly isForked: true

  /** Names of projections this one reads from (for tooling/visualization) */
  readonly reads: readonly string[]

  /** Signals this projection subscribes to (for tooling/visualization) */
  readonly signalSubscriptions: readonly SignalSubscription[]

  /** Context tag for dependency injection */
  readonly Tag: Context.Tag<ForkedProjectionInstance<TForkState>, ForkedProjectionInstance<TForkState>>

  /** Layer that provides the projection instance and signal PubSubs */
  readonly Layer: Layer.Layer<
    ForkedProjectionInstance<TForkState> | SignalPubSubs<AttachSourceState<TSignalDefs, ForkedState<TForkState>>>,
    never,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ProjectionBusService<TEvent> | PubSub.PubSub<any>
  >

  /** Signals with source state attached */
  readonly signals: AttachSourceState<TSignalDefs, ForkedState<TForkState>>
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

/**
 * Define a forked projection with per-fork state management.
 *
 * @typeParam TEvent - Event union type (must extend ForkableEvent)
 * @typeParam TForkState - State type for each fork
 *
 * @example
 * ```typescript
 * interface ForkMemoryState {
 *   messages: Message[]
 * }
 *
 * const MemoryProjection = Projection.defineForked<MyEvent, ForkMemoryState>()({
 *   name: 'Memory',
 *   initialFork: { messages: [] },
 *   reads: [SessionContextProjection],
 *
 *   eventHandlers: {
 *     user_message: ({ event, fork, read }) => {
 *       const ctx = read(SessionContextProjection)
 *       return {
 *         ...fork,
 *         messages: [...fork.messages, { role: 'user', content: event.content }]
 *       }
 *     }
 *   },
 *
 *   signalHandlers: (on) => [
 *     on(OtherProjection.signals.someSignal, ({ value, source, state, emit, read }) => {
 *       // Full type inference on all parameters
 *       return state
 *     })
 *   ]
 * })
 * ```
 */
export function defineForked<TEvent extends ForkableEvent, TForkState>() {
  return <
    TName extends string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
    TReads extends readonly AnyProjectionResult[] = readonly []
  >(
    config: ForkedProjectionConfig<TName, TForkState, TEvent, TSignalDefs, TReads>
  ): ForkedProjectionResult<TName, TForkState, TEvent, TSignalDefs> => {
    const serviceName = `${config.name}Projection`
    const Tag = Context.GenericTag<ForkedProjectionInstance<TForkState>>(serviceName)
    const BusTag = ProjectionBusTag<TEvent>()

    // Convert SignalDefs to Signals with source state attached
    type TFullState = ForkedState<TForkState>
    const signalDefs = config.signals ?? {} as TSignalDefs
    const signals = {} as Record<string, Signal<unknown, TFullState>>
    for (const [key, def] of Object.entries(signalDefs)) {
      signals[key] = fromDef<unknown, TFullState>(def as SignalDef<unknown>, config.name)
    }
    const typedSignals = signals as AttachSourceState<TSignalDefs, TFullState>

    // Build PubSub layers for Workers to subscribe (async)
    type TSignals = AttachSourceState<TSignalDefs, TFullState>
    const signalEntries = Object.entries(signals)

    let SignalPubSubLayers: Layer.Layer<SignalPubSubs<TSignals>, never, never> =
      Layer.empty as Layer.Layer<SignalPubSubs<TSignals>, never, never>

    for (const [, signal] of signalEntries) {
      const signalLayer = Layer.scoped(signal.tag, PubSub.unbounded<unknown>())
      SignalPubSubLayers = Layer.merge(SignalPubSubLayers, signalLayer) as Layer.Layer<SignalPubSubs<TSignals>, never, never>
    }

    // Extract read dependency names and track which are forked
    const readDeps = (config.reads ?? []) as readonly AnyProjectionResult[]
    const allowedReadNames = new Set(readDeps.map(p => p.name))
    const forkedReadNames = new Set(readDeps.filter(p => p.isForked).map(p => p.name))

    // Extract signal subscription metadata for tooling
    const signalSubscriptions: SignalSubscription[] = []
    if (config.signalHandlers) {
      const extractSignalName: ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads> = (signal) => {
        signalSubscriptions.push({
          signal: signal.name,
          source: signal.name.split('/')[0]
        })
        return { signal, handler: () => ({ forks: new Map() }) }
      }
      config.signalHandlers(extractSignalName)
    }

    const LogicLayer = Layer.scoped(
      Tag,
      Effect.gen(function* () {
        const bus = yield* BusTag

        // Initialize with root fork (null = root)
        const initialState: TFullState = {
          forks: new Map([[null, config.initialFork]])
        }
        const stateRef = yield* SubscriptionRef.make(initialState)

        // Register read dependencies with the bus
        for (const dep of readDeps) {
          yield* bus.registerDependency(config.name, dep.name)
        }

        // Register state getter for this projection
        yield* bus.registerStateGetter(
          config.name,
          () => Effect.runSync(SubscriptionRef.get(stateRef)),
          true // forked
        )

        // Build read function for event handlers (fork-aware)
        const makeEventReadFn = (forkId: string | null): ForkedReadFn<TReads> => {
          return <P extends TReads[number]>(projection: P): StateOfProjection<P> => {
            if (!allowedReadNames.has(projection.name)) {
              throw new Error(
                `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
              )
            }
            // For forked projections, resolve to the same forkId
            if (forkedReadNames.has(projection.name)) {
              return bus.getForkState(projection.name, forkId) as StateOfProjection<P>
            }
            return bus.getProjectionState(projection.name) as StateOfProjection<P>
          }
        }

        // Build read function for signal handlers (returns full state for forked deps)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const signalReadFn: ForkedSignalReadFn<TReads> = <P extends TReads[number]>(projection: P): any => {
          if (!allowedReadNames.has(projection.name)) {
            throw new Error(
              `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
            )
          }
          // Always return full state in signal handlers
          return bus.getProjectionState(projection.name)
        }

        // Get PubSubs for async worker subscriptions
        const pubsubs: Record<string, PubSub.PubSub<unknown>> = {}
        for (const [, signal] of signalEntries) {
          pubsubs[signal.name] = yield* signal.tag
        }

        // Pending signals
        let pendingSignalEffects: Effect.Effect<void>[] = []

        // Build sync emit functions
        const emitters: Record<string, (value: unknown) => void> = {}
        for (const [key, signal] of signalEntries) {
          const pubsub = pubsubs[signal.name]
          emitters[key] = (value: unknown) => {
            pendingSignalEffects.push(
              Effect.gen(function* () {
                const sourceState = yield* SubscriptionRef.get(stateRef)
                yield* bus.queueSignal(signal.name, value, sourceState)
                yield* PubSub.publish(pubsub, value)
              })
            )
          }
        }

        const typedEmitters = emitters as SignalEmitters<TSignalDefs>

        const flushPendingSignals = Effect.gen(function* () {
          const effects = pendingSignalEffects
          pendingSignalEffects = []
          for (const effect of effects) {
            yield* effect
          }
        })

        // Register signal handlers (operate on full state)
        if (config.signalHandlers) {
          // Create the `on` builder function
          const on: ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads> = (signal, handler) => ({
            signal,
            handler
          })

          // Call the builder to get the handler pairs
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
                    read: signalReadFn
                  })
                )
                yield* flushPendingSignals
              }),
              serviceName
            )
          }
        }

        // Build event handler - extracts forkId, manages Map
        const eventHandler = (event: TEvent): Effect.Effect<void> => {
          const handler = config.eventHandlers?.[event.type as TEvent['type']]
          const broadcastHandler = config.broadcastEventHandlers?.[event.type as TEvent['type']]
          if (!handler && !broadcastHandler) return Effect.void

          return Effect.gen(function* () {
            // Per-fork handler
            if (handler) {
              yield* SubscriptionRef.update(stateRef, (currentState) => {
                const forkId = event.forkId
                const currentFork = currentState.forks.get(forkId) ?? config.initialFork
                const readFn = makeEventReadFn(forkId)

                const newFork = handler({
                  event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                  fork: currentFork,
                  emit: typedEmitters,
                  read: readFn
                })

                const newForks = new Map(currentState.forks)
                // Fork cleanup: Event handlers can return null to remove fork from Map.
                // This prevents memory leaks for completed forks.
                if (newFork === null) {
                  newForks.delete(forkId)
                } else {
                  newForks.set(forkId, newFork)
                }
                return { forks: newForks }
              })
            }

            // Broadcast handlers - run for every fork
            if (broadcastHandler) {
              yield* SubscriptionRef.update(stateRef, (currentState) => {
                const newForks = new Map(currentState.forks)
                for (const [forkId, fork] of newForks) {
                  const readFn = makeEventReadFn(forkId)
                  const newFork = broadcastHandler({
                    event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                    forkId,
                    fork,
                    emit: typedEmitters,
                    read: readFn
                  })
                  if (newFork === null) {
                    newForks.delete(forkId)
                  } else {
                    newForks.set(forkId, newFork)
                  }
                }
                return { forks: newForks }
              })
            }

            yield* flushPendingSignals
          })
        }

        // Register with projection bus
        const eventTypes = [
          ...Object.keys(config.eventHandlers ?? {}),
          ...Object.keys(config.broadcastEventHandlers ?? {})
        ] as TEvent['type'][]

        yield* bus.register(eventHandler, eventTypes, serviceName)

        // Return instance with fork-aware accessors
        const instance: ForkedProjectionInstance<TForkState> = {
          getFork: (forkId) => Effect.gen(function* () {
            const state = yield* SubscriptionRef.get(stateRef)
            return state.forks.get(forkId) ?? config.initialFork
          }),
          getAllForks: () => Effect.gen(function* () {
            const state = yield* SubscriptionRef.get(stateRef)
            return state.forks
          }),
          state: stateRef
        }

        return instance
      })
    )

    // Compose layers
    type LayerOutput = ForkedProjectionInstance<TForkState> | SignalPubSubs<TSignals>
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    type LayerInput = ProjectionBusService<TEvent> | PubSub.PubSub<any>

    const FullLayer = Layer.provideMerge(LogicLayer, SignalPubSubLayers) as Layer.Layer<
      LayerOutput,
      never,
      LayerInput
    >

    return {
      name: config.name,
      isForked: true as const,
      reads: readDeps.map(p => p.name),
      signalSubscriptions,
      Tag,
      Layer: FullLayer,
      signals: typedSignals
    }
  }
}
