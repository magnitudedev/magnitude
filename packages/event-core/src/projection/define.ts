/**
 * Projection.define() - State derived from events with signal support
 *
 * Uses eventHandlers/signalHandlers pattern.
 * ALL communication (events and signals) is SYNCHRONOUS.
 *
 * Supports cross-projection reads via the `reads` config option.
 */

import { Effect, SubscriptionRef, Context, Layer, PubSub, Schema } from 'effect'
import type { ParseResult } from 'effect'
import type { EnforceJsonSafe } from '@magnitudedev/utils/schema'
import { ProjectionBusTag, type ProjectionBusService, type AddressedStateInfo } from '../core/projection-bus'
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
import { resolveProjectionHandlerResult, type ProjectionHandlerResult } from './handler-result'
import type { AddressedEntryStore } from '../addressed/entry-store'
import type { AddressedError } from '../addressed/errors'
import {
  makeProjectionAddressedRuntime,
  makeReadTracker,
  type AddressedConsumerState,
  type AddressedReadTracker,
  type ProjectionAddressedConsumers,
  type ProjectionAddressedDescriptors,
  type ProjectionAddressedHandles
} from './addressed'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ProjectionRestorePlan {
  readonly commit: Effect.Effect<void>
}

export interface ProjectionInstance<
  TStateSchema extends Schema.Schema.AnyNoContext,
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  readonly state: SubscriptionRef.SubscriptionRef<Schema.Schema.Type<TStateSchema>>
  readonly get: Effect.Effect<Schema.Schema.Type<TStateSchema>>
  readonly addressed: ProjectionAddressedConsumers<TAddressed>
  readonly snapshot: Effect.Effect<Schema.Schema.Encoded<TStateSchema>, ParseResult.ParseError | AddressedError>
  readonly prepareRestore: (snapshot: Schema.Schema.Encoded<TStateSchema>) => Effect.Effect<ProjectionRestorePlan, ParseResult.ParseError>
  readonly restore: (snapshot: Schema.Schema.Encoded<TStateSchema>) => Effect.Effect<void, ParseResult.ParseError>
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
 * Extract state type for non-forked ReadFn.
 * Non-forked projections reading forked projections get ForkedState<S>.
 * Non-forked projections reading non-forked projections get S.
 *
 * Addressed index fields appear as their consumer views (`readonly Item[]`
 * for sequences, plain objects for records) — the read() Proxies make this
 * true at runtime.
 */
type ExtractStateForNonForkedReader<I> =
  I extends ProjectionInstance<infer S, infer A>
    ? AddressedConsumerState<Schema.Schema.Type<S>, A>
    : I extends ForkedProjectionInstance<infer S, infer A>
      ? ForkedState<AddressedConsumerState<Schema.Schema.Type<S>, A>>
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
      ? I extends ProjectionInstance<infer S, infer A>
        ? AddressedConsumerState<Schema.Schema.Type<S>, A>
        : I extends ForkedProjectionInstance<infer S, infer A>
          ? AddressedConsumerState<Schema.Schema.Type<S>, A>
          : never
      : never
    : never

/** Union type for any projection result - uses any for event type to allow variance */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyProjectionResult =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ProjectionResult<string, Schema.Schema.AnyNoContext, any, any, any, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ForkedProjectionResult<string, any, any, any, any, any>

type ProjectionAddressedRequirement<TAddressed extends ProjectionAddressedDescriptors> =
  keyof TAddressed extends never ? never : AddressedEntryStore

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
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
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
      addressed: ProjectionAddressedHandles<TAddressed>
    }) => ProjectionHandlerResult<TState>
  ) => { signal: TSignal; handler: typeof handler }

export type ProjectionAmbientHandlerBuilder<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> =
  <C extends TAmbients[number]>(ambientDef: C, handler: (ctx: {
    value: AmbientValueOf<C>
    state: TState
    emit: SignalEmitters<TSignalDefs>
    read: ReadFn<TReads>
    ambient: AmbientReader<TAmbients>
    addressed: ProjectionAddressedHandles<TAddressed>
  }) => ProjectionHandlerResult<TState>) => { ambient: C; handler: typeof handler }

/**
 * Return type of signal handler builder - the { signal, handler } pair
 */
export type ProjectionSignalHandlerPair<
  TState,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> =
  ReturnType<ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients, TAddressed>>

export type ProjectionAmbientHandlerPair<
  TState,
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
          state: TState
          emit: SignalEmitters<TSignalDefs>
          read: ReadFn<TReads>
          ambient: AmbientReader<TAmbients>
          addressed: ProjectionAddressedHandles<TAddressed>
        }) => ProjectionHandlerResult<TState>
      } : never
    : never
}[number]

// ---------------------------------------------------------------------------
// Config and Result Types
// ---------------------------------------------------------------------------

export interface ProjectionConfig<
  TName extends string,
  TStateSchema extends Schema.Schema.AnyNoContext,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TReads extends readonly AnyProjectionResult[] = readonly [],
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  readonly name: TName
  readonly state: TStateSchema
  readonly initial: Schema.Schema.Type<TStateSchema>

  /** Signal definitions - will be transformed to Signal<T, TState> in the result */
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

  /** Event handlers - pure reducers that return new state */
  readonly eventHandlers?: {
    [E in TEvent['type']]?: (ctx: {
      event: Timestamped<Extract<TEvent, { type: E }>>
      state: Schema.Schema.Type<TStateSchema>
      emit: SignalEmitters<TSignalDefs>
      read: ReadFn<TReads>
      ambient: AmbientReader<TAmbients>
      addressed: ProjectionAddressedHandles<TAddressed>
    }) => ProjectionHandlerResult<Schema.Schema.Type<TStateSchema>>
  }

  /**
   * Signal handlers - subscribe to signals from other projections.
   * Use the builder pattern: `signalHandlers: (on) => [on(signal, handler)]`
   */
  readonly signalHandlers?: (
    on: ProjectionSignalHandlerBuilder<Schema.Schema.Type<TStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>
  ) => readonly ProjectionSignalHandlerPair<Schema.Schema.Type<TStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>[]

  /**
   * Ambient handlers - react to runtime ambient changes.
   * Use the builder pattern: `ambientHandlers: (on) => [on(ambientDef, handler)]`
   */
  readonly ambientHandlers?: (
    on: ProjectionAmbientHandlerBuilder<Schema.Schema.Type<TStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>
  ) => readonly ProjectionAmbientHandlerPair<Schema.Schema.Type<TStateSchema>, TSignalDefs, TReads, TAmbients, TAddressed>[]
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
  TStateSchema extends Schema.Schema.AnyNoContext,
  TEvent extends BaseEvent,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
  TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
  TAddressed extends ProjectionAddressedDescriptors = {}
> {
  /** Projection name - used for dependency graph */
  readonly name: TName
  readonly stateSchema: TStateSchema

  /** Marker to identify standard (non-forked) projections */
  readonly isForked: false

  /** Names of projections this one reads from (for tooling/visualization) */
  readonly reads: readonly string[]

  /** Ambients this projection reads from (for tooling/visualization and typing) */
  readonly ambients: TAmbients

  /** Signals this projection subscribes to (for tooling/visualization) */
  readonly signalSubscriptions: readonly SignalSubscription[]

  /** Context tag for dependency injection */
  readonly Tag: Context.Tag<
    ProjectionInstance<TStateSchema, TAddressed>,
    ProjectionInstance<TStateSchema, TAddressed>
  >

  /** Layer that provides the projection instance and signal PubSubs */
  readonly Layer: Layer.Layer<
    ProjectionInstance<TStateSchema, TAddressed> | SignalPubSubs<AttachSourceState<TSignalDefs, Schema.Schema.Type<TStateSchema>>>,
    AddressedError,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ProjectionBusService<TEvent> | PubSub.PubSub<any> | ProjectionAddressedRequirement<TAddressed>
  >

  /** Signals with source state attached - Signal<T, State> */
  readonly signals: AttachSourceState<TSignalDefs, Schema.Schema.Type<TStateSchema>>
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
export function define<TEvent extends BaseEvent>() {
  return <
    TName extends string,
    TStateSchema extends Schema.Schema.AnyNoContext,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    TSignalDefs extends Record<string, SignalDef<any>> = Record<string, never>,
    TReads extends readonly AnyProjectionResult[] = readonly [],
    TAmbients extends readonly AmbientDef<any, any>[] = readonly [],
    TAddressed extends ProjectionAddressedDescriptors = {}
  >(
    config: EnforceJsonSafe<TStateSchema, ProjectionConfig<TName, TStateSchema, TEvent, TSignalDefs, TReads, TAmbients, TAddressed>>
  ): ProjectionResult<TName, TStateSchema, TEvent, TSignalDefs, TAmbients, TAddressed> => {
    type TState = Schema.Schema.Type<TStateSchema>
    const serviceName = `${config.name}Projection`
    const Tag = Context.GenericTag<ProjectionInstance<TStateSchema, TAddressed>>(serviceName)
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
      const extractSignalName: ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients, TAddressed> = (signal) => {
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

        const addressedRuntime = yield* makeProjectionAddressedRuntime(
          config.name,
          (config.addressed ?? {}) as TAddressed,
          bus.notifyAddressedChange
        )
        let currentCommittedState = config.initial as TState
        const stateRef = yield* SubscriptionRef.make(currentCommittedState)
        const mutationLock = yield* Effect.makeSemaphore(1)

        // The active read tracker for the current handler invocation.
        // Created at handler start, read after handler completes.
        let activeReadTracker: AddressedReadTracker | null = null

        // Restore pins nothing: consumer pins re-establish on the first
        // rebuild via load-on-access, and owner writes load through the
        // transaction.
        const commitState = (state: TState): Effect.Effect<void> =>
          mutationLock.withPermits(1)(
            Effect.gen(function* () {
              yield* addressedRuntime.reset
              currentCommittedState = state
              yield* SubscriptionRef.set(stateRef, state)
            }).pipe(Effect.uninterruptible)
          )

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
          () => currentCommittedState,
          false // not forked
        )

        // Register addressed state info so consuming projections get Proxies
        if (!addressedRuntime.isEmpty) {
          const addressedInfo: AddressedStateInfo = {
            descriptors: addressedRuntime.descriptors,
            consumers: addressedRuntime.consumers,
            consumersForScope: addressedRuntime.consumersFor,
            pinConsumer: addressedRuntime.pinConsumer,
            isForked: false
          }
          yield* bus.registerAddressedState(config.name, addressedInfo)
        }

        // Build read function — uses tracker-aware state getter when a tracker is active
        const makeReadFn = (): ReadFn<TReads> => {
          return <P extends TReads[number]>(projection: P): StateOfProjection<P> => {
            if (!allowedReadNames.has(projection.name)) {
              throw new Error(
                `Projection "${config.name}" cannot read "${projection.name}" - not declared in reads`
              )
            }
            if (activeReadTracker) {
              return bus.getProjectionStateWithTracker(projection.name, activeReadTracker) as StateOfProjection<P>
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

        type PendingSignalEmit = {
          readonly signalName: string
          readonly value: unknown
          readonly pubsub: PubSub.PubSub<unknown>
        }

        let pendingSignalEmits: PendingSignalEmit[] = []

        // Build sync emit functions that queue signals to ProjectionBus
        // Also publishes to PubSub for workers (async subscribers)
        const emitters: Record<string, (value: unknown) => void> = {}
        for (const [key, signal] of signalEntries) {
          const pubsub = pubsubs[signal.name]
          emitters[key] = (value: unknown) => {
            pendingSignalEmits.push({ signalName: signal.name, value, pubsub })
          }
        }

        const typedEmitters = emitters as SignalEmitters<TSignalDefs>

        // Flush pending signal queue effects (runs queued signals to bus)
        const flushPendingSignals = (sourceState: TState) => Effect.gen(function* () {
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
          nextState: TState,
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
            state: TState,
            addressed: ProjectionAddressedHandles<TAddressed>
          ) => ProjectionHandlerResult<TState>
        ): Effect.Effect<void, unknown> =>
          mutationLock.withPermits(1)(
            Effect.gen(function* () {
              // Create a fresh read tracker for this handler invocation
              activeReadTracker = makeReadTracker()
              const addressedResult = yield* addressedRuntime.transact((addressedMutation) =>
                resolveProjectionHandlerResult(
                  update(currentCommittedState, addressedMutation.handles)
                )
              )
              yield* commitMutation(addressedResult.value, addressedResult.changed)
            }).pipe(
              Effect.ensuring(Effect.sync(() => { activeReadTracker = null })),
              Effect.ensuring(discardPendingSignals)
            )
          )

        // Register signal handlers with the bus (SYNC dispatch during flush phase)
        if (config.signalHandlers) {
          const on: ProjectionSignalHandlerBuilder<TState, TSignalDefs, TReads, TAmbients, TAddressed> = (signal, handler) => ({
            signal,
            handler
          })
          const handlerPairs = config.signalHandlers(on)

          for (const { signal, handler } of handlerPairs) {
            yield* bus.registerSignalHandler(
              signal.name,
              (value, sourceState) =>
                mutateState((currentState, addressed) =>
                  handler({
                    value,
                    source: sourceState,
                    state: currentState,
                    emit: typedEmitters,
                    read: readFn,
                    ambient: ambientReader,
                    addressed
                  })
                ),
              serviceName
            )
          }
        }

        if (config.ambientHandlers) {
          const on: ProjectionAmbientHandlerBuilder<TState, TSignalDefs, TReads, TAmbients, TAddressed> = (ambientDef, handler) => ({
            ambient: ambientDef,
            handler
          })
          const handlerPairs = config.ambientHandlers(on)
          const registerAmbientPair = <C extends TAmbients[number]>(pair: {
            readonly ambient: C
            readonly handler: (ctx: {
              readonly value: AmbientValueOf<C>
              readonly state: TState
              readonly emit: SignalEmitters<TSignalDefs>
              readonly read: ReadFn<TReads>
              readonly ambient: AmbientReader<TAmbients>
              readonly addressed: ProjectionAddressedHandles<TAddressed>
            }) => ProjectionHandlerResult<TState>
          }) =>
            bus.registerAmbientHandler(
              pair.ambient.name,
              (value) =>
                mutateState((currentState, addressed) =>
                  pair.handler({
                    value: value as AmbientValueOf<C>,
                    state: currentState,
                    emit: typedEmitters,
                    read: readFn,
                    ambient: ambientReader,
                    addressed
                  })
                ),
              serviceName
            )

          for (const pair of handlerPairs) {
            yield* registerAmbientPair(pair)
          }
        }

        // Build combined event handler from individual handlers
        const eventHandler = (event: TEvent): Effect.Effect<void, unknown> => {
          if (!config.eventHandlers) return Effect.void

          const handler = config.eventHandlers[event.type as TEvent['type']]
          if (!handler) return Effect.void

          return mutateState((currentState, addressed) =>
            handler({
              event: event as Timestamped<Extract<TEvent, { type: typeof event.type }>>,
              state: currentState,
              emit: typedEmitters,
              read: readFn,
              ambient: ambientReader,
              addressed
            })
          )
        }

        // Register with projection bus
        // Registration order follows layer build order, which respects signal dependencies
        const eventTypes = config.eventHandlers
          ? Object.keys(config.eventHandlers) as TEvent['type'][]
          : []

        yield* bus.register(eventHandler, eventTypes, serviceName)

        const prepareRestore = (snapshot: Schema.Schema.Encoded<TStateSchema>) =>
          Effect.map(
            Schema.decode(config.state)(snapshot),
            (state): ProjectionRestorePlan => ({
              commit: commitState(state)
            })
          )

        return {
          state: stateRef,
          get: SubscriptionRef.get(stateRef),
          addressed: addressedRuntime.consumers,
          snapshot: mutationLock.withPermits(1)(Effect.gen(function* () {
            yield* addressedRuntime.flushDirty
            const state = yield* SubscriptionRef.get(stateRef)
            return yield* Schema.encode(config.state)(state)
          })),
          prepareRestore,
          restore: (snapshot) => Effect.gen(function* () {
            const plan = yield* prepareRestore(snapshot)
            yield* plan.commit
          })
        }
      })
    )

    // Compose layers
    type LayerOutput = ProjectionInstance<TStateSchema, TAddressed> | SignalPubSubs<TSignals>
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
      stateSchema: config.state,
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
