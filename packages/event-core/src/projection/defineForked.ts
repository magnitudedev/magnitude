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

import { Effect, SubscriptionRef, Context, Layer, PubSub, Schema } from 'effect'
import type { ParseResult } from 'effect'
import type { EnforceJsonSafe } from '@magnitudedev/utils/schema'
import { ProjectionBusTag, type ProjectionBusService, type AddressedStateInfo } from '../core/projection-bus'
import { AmbientServiceTag } from '../core/ambient-service'
import { type BaseEvent, type Timestamped } from '../core/event-bus-core'
import type { AddressedEntryStore } from '../addressed/entry-store'
import type { AddressedError } from '../addressed/errors'
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
  type ProjectionRestorePlan,
  type StateOfProjection,
  type StateOfProjectionForForkedReader,
  type SignalSubscription,
  type AmbientReader
} from './define'
import { type AmbientDef, type AmbientValueOf } from '../ambient/define'
import { resolveProjectionHandlerResult, type ProjectionHandlerResult } from './handler-result'
import {
  makeProjectionAddressedRuntime,
  makeReadTracker,
  type AddressedConsumerState,
  type AddressedReadTracker,
  type ProjectionAddressedDescriptors,
  type ProjectionForkedAddressedHandles,
  type ProjectionForkedAddressedConsumers,
  type ProjectionAddressedHandles,
  type ProjectionAddressedMutation
} from './addressed'

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
export type ForkedProjectionSnapshot<TForkStateSchema extends Schema.Schema.AnyNoContext> =
  readonly (readonly [string | null, Schema.Schema.Encoded<TForkStateSchema>])[]

export interface ForkedProjectionInstance<
  TForkStateSchema extends Schema.Schema.AnyNoContext,
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  /** Get state for a specific fork */
  readonly getFork: (forkId: string | null) => Effect.Effect<Schema.Schema.Type<TForkStateSchema>>
  /** Get all forks */
  readonly getAllForks: () => Effect.Effect<Map<string | null, Schema.Schema.Type<TForkStateSchema>>>
  /** Addressed collections scoped by fork. */
  readonly addressed: ProjectionForkedAddressedConsumers<TAddressed>
  /** Raw state ref (for subscriptions) */
  readonly state: SubscriptionRef.SubscriptionRef<ForkedState<Schema.Schema.Type<TForkStateSchema>>>
  readonly snapshot: Effect.Effect<ForkedProjectionSnapshot<TForkStateSchema>, ParseResult.ParseError | AddressedError>
  readonly prepareRestore: (snapshot: ForkedProjectionSnapshot<TForkStateSchema>) => Effect.Effect<ProjectionRestorePlan, ParseResult.ParseError>
  readonly restore: (snapshot: ForkedProjectionSnapshot<TForkStateSchema>) => Effect.Effect<void, ParseResult.ParseError>
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
export type ForkedReadFn<TReads extends readonly AnyProjectionResult[]> = {
  <P extends TReads[number]>(projection: P): StateOfProjectionForForkedReader<P>
  <P extends TReads[number]>(
    projection: P,
    forkId: string | null
  ): P extends ForkedProjectionResult<any, infer TForkStateSchema, any, any, any, infer TAddressed>
    ? AddressedConsumerState<Schema.Schema.Type<TForkStateSchema>, TAddressed>
    : StateOfProjectionForForkedReader<P>
}

/**
 * Read function for signal handlers in forked projections.
 * For forked dependencies, returns the full ForkedState since there's no single forkId context.
 */
export type ForkedSignalReadFn<TReads extends readonly AnyProjectionResult[]> =
  <P extends TReads[number]>(projection: P) =>
    P extends ForkedProjectionResult<any, infer TForkStateSchema, any, any, any, infer TAddressed>
      ? ForkedState<AddressedConsumerState<Schema.Schema.Type<TForkStateSchema>, TAddressed>>
      : StateOfProjectionForForkedReader<P>

type ProjectionAddressedRequirement<TAddressed extends ProjectionAddressedDescriptors> =
  keyof TAddressed extends never ? never : AddressedEntryStore

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
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
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
      ambient: AmbientReader<TAmbients>
      addressed: ProjectionForkedAddressedHandles<TAddressed>
    }) => ProjectionHandlerResult<ForkedState<TForkState>>
  ) => { signal: TSignal; handler: typeof handler }

export type ForkedAmbientHandlerBuilder<
  TForkState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> =
  <C extends TAmbients[number]>(ambientDef: C, handler: (ctx: {
    value: AmbientValueOf<C>
    state: ForkedState<TForkState>
    emit: SignalEmitters<TSignalDefs>
    read: ForkedSignalReadFn<TReads>
    ambient: AmbientReader<TAmbients>
    addressed: ProjectionForkedAddressedHandles<TAddressed>
  }) => ProjectionHandlerResult<ForkedState<TForkState>>) => { ambient: C; handler: typeof handler }

/**
 * Return type of signal handler builder - the { signal, handler } pair
 */
export type ForkedSignalHandlerPair<
  TForkState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> =
  ReturnType<ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads, TAmbients, TAddressed>>

export type ForkedAmbientHandlerPair<
  TForkState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> = {
  [I in keyof TAmbients]: TAmbients[I] extends infer C
    ? C extends AmbientDef<infer _Value, infer _Requirements> ? {
        ambient: C
        handler: (ctx: {
          value: AmbientValueOf<C>
          state: ForkedState<TForkState>
          emit: SignalEmitters<TSignalDefs>
          read: ForkedSignalReadFn<TReads>
          ambient: AmbientReader<TAmbients>
          addressed: ProjectionForkedAddressedHandles<TAddressed>
        }) => ProjectionHandlerResult<ForkedState<TForkState>>
      } : never
    : never
}[number]

// ---------------------------------------------------------------------------
// Config and Result Types
// ---------------------------------------------------------------------------

export interface ForkedProjectionConfig<
  TName extends string,
  TForkStateSchema extends Schema.Schema.AnyNoContext,
  TEvent extends ForkableEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  readonly name: TName

  /** Initial state for each fork (used when a new forkId is first seen) */
  readonly initialFork: Schema.Schema.Type<TForkStateSchema>
  readonly forkState: TForkStateSchema

  /** Signal definitions */
  readonly signals?: TSignalDefs

  /**
   * Projections that this projection can read from.
   * Creates dependency edges in the execution graph.
   */
  readonly reads?: TReads

  /** Ambient dependencies that this projection can read from synchronously. */
  readonly ambients?: TAmbients

  /** Addressed collections owned by this projection. */
  readonly addressed?: TAddressed

  /**
   * Event handlers - receive fork state, return new fork state.
   * Framework extracts forkId from event, manages the Map.
   * 
   * Return null to delete the fork from the Map (useful for cleanup on fork_completed).
   */
  readonly eventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      fork: Schema.Schema.Type<TForkStateSchema>
      emit: SignalEmitters<TSignalDefs>
      read: ForkedReadFn<TReads>
      ambient: AmbientReader<TAmbients>
      addressed: ProjectionAddressedHandles<TAddressed>
    }) => ProjectionHandlerResult<Schema.Schema.Type<TForkStateSchema> | null>
  }

  /**
   * Global event handlers - receive full forked state for cross-fork updates.
   * Runs after per-fork eventHandlers and before signal flush.
   */
  readonly globalEventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      state: ForkedState<Schema.Schema.Type<TForkStateSchema>>
      emit: SignalEmitters<TSignalDefs>
      read: ForkedSignalReadFn<TReads>
      ambient: AmbientReader<TAmbients>
      addressed: ProjectionForkedAddressedHandles<TAddressed>
    }) => ProjectionHandlerResult<ForkedState<Schema.Schema.Type<TForkStateSchema>>>
  }

  /**
   * Signal handlers - subscribe to signals from other projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: ForkedSignalHandlerBuilder<Schema.Schema.Type<TForkStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>
  ) => readonly ForkedSignalHandlerPair<Schema.Schema.Type<TForkStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>[]

  /**
   * Ambient handlers - react to runtime ambient changes.
   * Use the builder pattern: `ambientHandlers: (on) => [on(ambientDef, handler)]`
   */
  readonly ambientHandlers?: (
    on: ForkedAmbientHandlerBuilder<Schema.Schema.Type<TForkStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>
  ) => readonly ForkedAmbientHandlerPair<Schema.Schema.Type<TForkStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>[]
}

export interface ForkedProjectionResult<
  TName extends string,
  TForkStateSchema extends Schema.Schema.AnyNoContext,
  TEvent extends ForkableEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  readonly name: TName
  readonly forkStateSchema: TForkStateSchema

  /** Marker to identify forked projections */
  readonly isForked: true

  /** Names of projections this one reads from (for tooling/visualization) */
  readonly reads: readonly string[]

  /** Ambients this projection reads from (for tooling/visualization and typing) */
  readonly ambients: TAmbients

  /** Signals this projection subscribes to (for tooling/visualization) */
  readonly signalSubscriptions: readonly SignalSubscription[]

  /** Context tag for dependency injection */
  readonly Tag: Context.Tag<
    ForkedProjectionInstance<TForkStateSchema, TAddressed>,
    ForkedProjectionInstance<TForkStateSchema, TAddressed>
  >

  /** Layer that provides the projection instance and signal PubSubs */
  readonly Layer: Layer.Layer<
    ForkedProjectionInstance<TForkStateSchema, TAddressed> | SignalPubSubs<AttachSourceState<TSignalDefs, ForkedState<Schema.Schema.Type<TForkStateSchema>>>>,
    AddressedError,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ProjectionBusService<TEvent> | PubSub.PubSub<any> | ProjectionAddressedRequirement<TAddressed>
  >

  /** Signals with source state attached */
  readonly signals: AttachSourceState<TSignalDefs, ForkedState<Schema.Schema.Type<TForkStateSchema>>>
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
 * interface ForkWindowState {
 *   messages: Message[]
 * }
 *
 * const WindowProjection = Projection.defineForked<MyEvent, ForkWindowState>()({
 *   name: 'Window',
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
export function defineForked<TEvent extends ForkableEvent>() {
  return <
    TName extends string,
    TForkStateSchema extends Schema.Schema.AnyNoContext,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
    TReads extends readonly AnyProjectionResult[] = readonly [],
    TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
    TAddressed extends ProjectionAddressedDescriptors = {}
  >(
    config: EnforceJsonSafe<TForkStateSchema, ForkedProjectionConfig<TName, TForkStateSchema, TEvent, TSignalDefs, TReads, TAmbients, TAddressed>>
  ): ForkedProjectionResult<TName, TForkStateSchema, TEvent, TSignalDefs, TAmbients, TAddressed> => {
    type TForkState = Schema.Schema.Type<TForkStateSchema>
    const serviceName = `${config.name}Projection`
    const Tag = Context.GenericTag<ForkedProjectionInstance<TForkStateSchema, TAddressed>>(serviceName)
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
    const ambients = (config.ambients ?? []) as TAmbients
    const allowedReadNames = new Set(readDeps.map(p => p.name))
    const forkedReadNames = new Set(readDeps.filter(p => p.isForked).map(p => p.name))

    // Extract signal subscription metadata for tooling
    const signalSubscriptions: SignalSubscription[] = []
    if (config.signalHandlers) {
      const extractSignalName: ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads, TAmbients, TAddressed> = (signal) => {
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
        const ambientService = yield* AmbientServiceTag

        const addressedRuntime = yield* makeProjectionAddressedRuntime(
          config.name,
          (config.addressed ?? {}) as TAddressed,
          bus.notifyAddressedChange
        )

        for (const ambientDef of ambients) {
          yield* ambientService.register(ambientDef)
        }

        // Initialize with root fork (null = root)
        const initialState: TFullState = {
          forks: new Map([[null, config.initialFork as TForkState]])
        }
        let currentCommittedState = initialState
        const stateRef = yield* SubscriptionRef.make(initialState)
        const mutationLock = yield* Effect.makeSemaphore(1)

        // The active read tracker for the current handler invocation.
        let activeReadTracker: AddressedReadTracker | null = null

        // Restore pins nothing: consumer pins re-establish on the first
        // rebuild via load-on-access, and owner writes load through the
        // transaction.
        const commitState = (state: TFullState): Effect.Effect<void> =>
          mutationLock.withPermits(1)(
            Effect.gen(function* () {
              yield* addressedRuntime.reset
              currentCommittedState = state
              yield* SubscriptionRef.set(stateRef, state)
            }).pipe(Effect.uninterruptible)
          )

        // Register read dependencies with the bus
        for (const dep of readDeps) {
          yield* bus.registerDependency(config.name, dep.name)
        }

        // Register state getter for this projection
        yield* bus.registerStateGetter(
          config.name,
          () => currentCommittedState,
          true // forked
        )

        // Register addressed state info so consuming projections get Proxies
        if (!addressedRuntime.isEmpty) {
          const addressedInfo: AddressedStateInfo = {
            descriptors: addressedRuntime.descriptors,
            consumers: addressedRuntime.consumers,
            consumersForScope: addressedRuntime.consumersFor,
            pinConsumer: addressedRuntime.pinConsumer,
            isForked: true
          }
          yield* bus.registerAddressedState(config.name, addressedInfo)
        }

        // Build read function for event handlers (fork-aware)
        const makeEventReadFn = (eventForkId: string | null): ForkedReadFn<TReads> => {
          return <P extends TReads[number]>(projection: P, forkId?: string | null): StateOfProjectionForForkedReader<P> => {
            if (!allowedReadNames.has(projection.name)) {
              throw new Error(
                `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
              )
            }
            // For forked projections, resolve to specified forkId or default to event forkId
            if (forkedReadNames.has(projection.name)) {
              const targetForkId = forkId !== undefined ? forkId : eventForkId
              if (activeReadTracker) {
                return bus.getForkStateWithTracker(projection.name, targetForkId, activeReadTracker) as StateOfProjectionForForkedReader<P>
              }
              return bus.getForkState(projection.name, targetForkId) as StateOfProjectionForForkedReader<P>
            }
            if (activeReadTracker) {
              return bus.getProjectionStateWithTracker(projection.name, activeReadTracker) as StateOfProjectionForForkedReader<P>
            }
            return bus.getProjectionState(projection.name) as StateOfProjectionForForkedReader<P>
          }
        }

        // Build read function for signal handlers (returns full state for forked deps)
        const signalReadFn: ForkedSignalReadFn<TReads> = <P extends TReads[number]>(projection: P) => {
          if (!allowedReadNames.has(projection.name)) {
            throw new Error(
              `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
            )
          }
          // Always return full state in signal handlers
          if (activeReadTracker) {
            return bus.getProjectionStateWithTracker(projection.name, activeReadTracker) as ReturnType<ForkedSignalReadFn<TReads>>
          }
          return bus.getProjectionState(projection.name) as ReturnType<ForkedSignalReadFn<TReads>>
        }

        const ambientReader: AmbientReader<TAmbients> = {
          get: <C extends TAmbients[number]>(def: C): AmbientValueOf<C> => ambientService.getValue(def)
        }

        // Get PubSubs for async worker subscriptions
        const pubsubs: Record<string, PubSub.PubSub<unknown>> = {}
        for (const [, signal] of signalEntries) {
          pubsubs[signal.name] = yield* signal.tag
        }

        type PendingSignalEmit = {
          readonly signalName: string
          readonly value: unknown
          readonly pubsub: PubSub.PubSub<unknown>
        }

        let pendingSignalEmits: PendingSignalEmit[] = []

        // Build sync emit functions
        const emitters: Record<string, (value: unknown) => void> = {}
        for (const [key, signal] of signalEntries) {
          const pubsub = pubsubs[signal.name]
          emitters[key] = (value: unknown) => {
            pendingSignalEmits.push({ signalName: signal.name, value, pubsub })
          }
        }

        const typedEmitters = emitters as SignalEmitters<TSignalDefs>

        const flushPendingSignals = (sourceState: TFullState) => Effect.gen(function* () {
          const emits = pendingSignalEmits
          pendingSignalEmits = []
          for (const emit of emits) {
            yield* bus.queueSignal(emit.signalName, emit.value, sourceState)
            yield* PubSub.publish(emit.pubsub, emit.value)
          }
        })

        const discardPendingSignals = Effect.sync(() => {
          pendingSignalEmits = []
        })

        const commitMutation = (
          nextState: TFullState,
          changedAddresses: ReadonlyMap<string, ReadonlySet<string>>
        ): Effect.Effect<void, unknown> =>
          Effect.gen(function* () {
            currentCommittedState = nextState
            yield* SubscriptionRef.set(stateRef, nextState)
            yield* flushPendingSignals(nextState)
            yield* addressedRuntime.publish(changedAddresses)
            // For each (source, property) this handler read through Proxies:
            // record the tracked set for addressed-change triggering, and pin
            // it in the source's space so the segments stay resident. Entries
            // not touched this invocation keep their previous tracked set —
            // state derived from earlier reads is still live.
            if (activeReadTracker) {
              for (const [source, perProperty] of activeReadTracker) {
                for (const [property, addresses] of perProperty) {
                  bus.updateAddressedDependencies(serviceName, source, property, addresses)
                  yield* bus.pinAddressedConsumer(source, property, `consumer:${serviceName}`, addresses)
                }
              }
            }
          }).pipe(Effect.uninterruptible)

        const mutateState = (
          update: (
            state: TFullState,
            addressedMutation: ProjectionAddressedMutation<TAddressed>
          ) => ProjectionHandlerResult<TFullState>
        ): Effect.Effect<void, unknown> =>
          mutationLock.withPermits(1)(
            Effect.gen(function* () {
              // Create a fresh read tracker for this handler invocation
              activeReadTracker = makeReadTracker()
              const addressedResult = yield* addressedRuntime.transact((addressedMutation) =>
                resolveProjectionHandlerResult(
                  update(currentCommittedState, addressedMutation)
                )
              )
              yield* commitMutation(addressedResult.value, addressedResult.changed)
            }).pipe(
              Effect.ensuring(Effect.sync(() => { activeReadTracker = null })),
              Effect.ensuring(discardPendingSignals)
            )
          )

        // Register signal handlers (operate on full state)
        if (config.signalHandlers) {
          // Create the `on` builder function
          const on: ForkedSignalHandlerBuilder<TForkState, TSignalDefs, TReads, TAmbients, TAddressed> = (signal, handler) => ({
            signal,
            handler
          })

          // Call the builder to get the handler pairs
          const handlerPairs = config.signalHandlers(on)
          const forkedAddressed = (
            mutation: ProjectionAddressedMutation<TAddressed>
          ): ProjectionForkedAddressedHandles<TAddressed> => ({
            forFork: (forkId) => mutation.handlesFor(['forks', forkId ?? 'root'])
          })

          for (const { signal, handler } of handlerPairs) {
            yield* bus.registerSignalHandler(
              signal.name,
              (value, sourceState) =>
                mutateState((currentState, addressedMutation) =>
                  handler({
                    value,
                    source: sourceState,
                    state: currentState,
                    emit: typedEmitters,
                    read: signalReadFn,
                    ambient: ambientReader,
                    addressed: forkedAddressed(addressedMutation)
                  })
                ),
              serviceName
            )
          }
        }

        if (config.ambientHandlers) {
          const on: ForkedAmbientHandlerBuilder<TForkState, TSignalDefs, TReads, TAmbients, TAddressed> = (ambientDef, handler) => ({
            ambient: ambientDef,
            handler
          })
          const handlerPairs = config.ambientHandlers(on)
          const forkedAddressed = (
            mutation: ProjectionAddressedMutation<TAddressed>
          ): ProjectionForkedAddressedHandles<TAddressed> => ({
            forFork: (forkId) => mutation.handlesFor(['forks', forkId ?? 'root'])
          })
          const registerAmbientPair = <C extends TAmbients[number]>(pair: {
            readonly ambient: C
            readonly handler: (ctx: {
              readonly value: AmbientValueOf<C>
              readonly state: ForkedState<TForkState>
              readonly emit: SignalEmitters<TSignalDefs>
              readonly read: ForkedSignalReadFn<TReads>
              readonly ambient: AmbientReader<TAmbients>
              readonly addressed: ProjectionForkedAddressedHandles<TAddressed>
            }) => ProjectionHandlerResult<ForkedState<TForkState>>
          }) =>
            bus.registerAmbientHandler(
              pair.ambient.name,
              (value) =>
                mutateState((currentState, addressedMutation) =>
                  pair.handler({
                    value: value as AmbientValueOf<C>,
                    state: currentState,
                    emit: typedEmitters,
                    read: signalReadFn,
                    ambient: ambientReader,
                    addressed: forkedAddressed(addressedMutation)
                  })
                ),
              serviceName
            )

          for (const pair of handlerPairs) {
            yield* registerAmbientPair(pair)
          }
        }

        // Build event handler - extracts forkId, manages Map
        const eventHandler = (event: TEvent): Effect.Effect<void, unknown> => {
          const handler = config.eventHandlers?.[event.type as TEvent['type']]
          const globalHandler = config.globalEventHandlers?.[event.type as TEvent['type']]
          if (!handler && !globalHandler) return Effect.void

          return mutateState((currentState, addressedMutation) => Effect.gen(function* () {
            let nextState = currentState

            // Per-fork handler
            if (handler) {
              const forkId = event.forkId
              const currentFork = nextState.forks.get(forkId) ?? config.initialFork
              const readFn = makeEventReadFn(forkId)
              const addressed = addressedMutation.handlesFor(['forks', forkId ?? 'root'])

              const newFork = yield* resolveProjectionHandlerResult(handler({
                event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                fork: currentFork,
                emit: typedEmitters,
                read: readFn,
                ambient: ambientReader,
                addressed
              }))

              const newForks = new Map(nextState.forks)
              // Fork cleanup: Event handlers can return null to remove fork from Map.
              // This prevents memory leaks for completed forks.
              if (newFork === null) {
                newForks.delete(forkId)
              } else {
                newForks.set(forkId, newFork)
              }
              nextState = { forks: newForks }
            }

            // Global event handler - full state access
            if (globalHandler) {
              const addressed: ProjectionForkedAddressedHandles<TAddressed> = {
                forFork: (forkId) => addressedMutation.handlesFor(['forks', forkId ?? 'root'])
              }
              nextState = yield* resolveProjectionHandlerResult(globalHandler({
                event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
                state: nextState,
                emit: typedEmitters,
                read: signalReadFn,
                ambient: ambientReader,
                addressed
              }))
            }

            return nextState
          }))
        }

        // Register with projection bus
        const eventTypes = [
          ...Object.keys(config.eventHandlers ?? {}),
          ...Object.keys(config.globalEventHandlers ?? {})
        ] as TEvent['type'][]

        yield* bus.register(eventHandler, eventTypes, serviceName)

        const prepareRestore = (snapshot: ForkedProjectionSnapshot<TForkStateSchema>) => Effect.gen(function* () {
          const forks = new Map<string | null, TForkState>()
          for (const entry of snapshot) {
            const [forkId, encodedFork] = entry
            forks.set(forkId, yield* Schema.decode(config.forkState)(encodedFork))
          }
          const restoredState: TFullState = { forks }
          return {
            commit: commitState(restoredState)
          } satisfies ProjectionRestorePlan
        })

        // Return instance with fork-aware accessors
        const instance: ForkedProjectionInstance<TForkStateSchema, TAddressed> = {
          getFork: (forkId) => Effect.gen(function* () {
            const state = yield* SubscriptionRef.get(stateRef)
            return state.forks.get(forkId) ?? config.initialFork
          }),
          getAllForks: () => Effect.gen(function* () {
            const state = yield* SubscriptionRef.get(stateRef)
            return state.forks
          }),
          addressed: {
            ...addressedRuntime.consumers,
            forFork: (forkId: string | null) => addressedRuntime.consumersFor(['forks', forkId ?? 'root'])
          } as ProjectionForkedAddressedConsumers<TAddressed>,
          state: stateRef,
          snapshot: mutationLock.withPermits(1)(Effect.gen(function* () {
            yield* addressedRuntime.flushDirty
            const state = yield* SubscriptionRef.get(stateRef)
            const entries: Array<readonly [string | null, Schema.Schema.Encoded<TForkStateSchema>]> = []
            for (const [forkId, fork] of state.forks) {
              entries.push([forkId, yield* Schema.encode(config.forkState)(fork)])
            }
            return entries
          })),
          prepareRestore,
          restore: (snapshot) => Effect.gen(function* () {
            const plan = yield* prepareRestore(snapshot)
            yield* plan.commit
          })
        }

        return instance
      })
    )

    // Compose layers
    type LayerOutput = ForkedProjectionInstance<TForkStateSchema, TAddressed> | SignalPubSubs<TSignals>
    type LayerInput =
      | ProjectionBusService<TEvent>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      | PubSub.PubSub<any>
      | ProjectionAddressedRequirement<TAddressed>

    const FullLayer = Layer.provideMerge(LogicLayer, SignalPubSubLayers) as Layer.Layer<
      LayerOutput,
      never,
      LayerInput
    >

    return {
      name: config.name,
      forkStateSchema: config.forkState,
      isForked: true as const,
      reads: readDeps.map(p => p.name),
      ambients,
      signalSubscriptions,
      Tag,
      Layer: FullLayer,
      signals: typedSignals
    }
  }
}
