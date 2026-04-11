import { Effect, Layer, ManagedRuntime, Stream, SubscriptionRef, PubSub, Context, Cause } from 'effect'
import { makeEventBusCoreLayer, type BaseEvent, type EventBusCoreService } from '../core/event-bus-core'
import { EventSinkTag, makeEventSinkLayer, type EventSinkService } from '../core/event-sink'
import { HydrationContext } from '../core/hydration-context'
import { InterruptCoordinatorLive } from '../core/interrupt-coordinator'
import { makeProjectionBusLayer, ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
import { makeWorkerBusLayer, WorkerBusTag, type WorkerBusService } from '../core/worker-bus'
import { type Signal } from '../signal/define'
import {
  FrameworkErrorPubSub,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporter,
  FrameworkErrorReporterLive,
  FrameworkError,
  type FrameworkErrorReporterService
} from '../core/framework-error'

// ---------------------------------------------------------------------------
// Core Services
// ---------------------------------------------------------------------------

export type CoreServices<TEvent extends BaseEvent> =
  | HydrationContext
  | EventSinkService<TEvent>
  | EventBusCoreService<TEvent>
  | ProjectionBusService<TEvent>
  | WorkerBusService<TEvent>
  | FrameworkErrorReporterService
  | PubSub.PubSub<FrameworkError>

// ---------------------------------------------------------------------------
// Component Interfaces
// ---------------------------------------------------------------------------

/**
 * Projection component - uses `any` for variance bypass during acceptance.
 * Type extraction uses conditional types with `infer` to recover actual types.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface ProjectionComponent {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly Layer: Layer.Layer<any, never, any>
}

/**
 * Worker component - workers output void and have requirements R.
 * Uses `any` for requirements to allow variance bypass.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface WorkerComponent {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly Layer: Layer.Layer<void, never, any>
}

/** Extract requirements from a worker's Layer */
type ExtractWorkerRequirements<W> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  W extends { readonly Layer: Layer.Layer<infer _A, infer _E, infer R> } ? R : never

/** Extract all worker requirements as a union */
type ExtractAllWorkerRequirements<T extends readonly WorkerComponent[]> =
  ExtractWorkerRequirements<T[number]>

/**
 * Internal requirements that the agent provides:
 * - CoreServices (bus, hydration, etc.)
 * - Projection outputs (signals, state)
 * - PubSub for any signal type
 */
type InternalRequirements<TEvent extends BaseEvent, TProjections extends readonly ProjectionComponent[]> =
  | CoreServices<TEvent>
  | ExtractProjectionOutputs<TProjections>
  | PubSub.PubSub<unknown>  // Signal PubSubs are provided internally

/**
 * External requirements = Worker requirements minus internal requirements.
 * These must be provided by the user when creating the agent client.
 */
type ExtractExternalRequirements<
  TEvent extends BaseEvent,
  TProjections extends readonly ProjectionComponent[],
  TWorkers extends readonly WorkerComponent[]
> = Exclude<
  ExtractAllWorkerRequirements<TWorkers>,
  InternalRequirements<TEvent, TProjections>
>

type ExtractLayerOutput<L> = L extends Layer.Layer<infer Out, infer _E, infer _R> ? Out : never
type ExtractProjectionOutputs<T extends readonly ProjectionComponent[]> =
  ExtractLayerOutput<T[number]['Layer']>

// ---------------------------------------------------------------------------
// Type Extractors for Projections
// ---------------------------------------------------------------------------

/**
 * A standard projection that has a Tag for state access
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface StandardProjection {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly Tag: Context.Tag<any, any>
  readonly isForked: false
}

/**
 * A forked projection with per-fork state
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
interface ForkedProjection {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly Tag: Context.Tag<any, any>
  readonly isForked: true
}

/**
 * A projection that has a Tag for state access (standard or forked)
 */
type ProjectionWithTag = StandardProjection | ForkedProjection

/**
 * Extract all signals from a projection.
 * Uses 'any' to match Signal<T> regardless of T's variance.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ExtractSignalsFromProjection<T> = T extends { readonly signals: infer S } ? S : never

/**
 * Get all signal values from projections as a union
 */
type AllSignalValues<T extends readonly ProjectionComponent[]> =
  ExtractSignalsFromProjection<T[number]> extends infer S
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ? S extends Record<string, Signal<any>>
      ? S[keyof S]
      : never
    : never

/**
 * Extract projections that have Tags (for state access)
 */
type ProjectionsWithTags<T extends readonly ProjectionComponent[]> =
  Extract<T[number], ProjectionWithTag>

// ---------------------------------------------------------------------------
// Expose Config Types - constrained to what projections provide
// ---------------------------------------------------------------------------

/**
 * Valid signals config - must be a subset of signals provided by projections
 */
type ValidSignalsConfig<TProjections extends readonly ProjectionComponent[]> =
  Record<string, AllSignalValues<TProjections>>

/**
 * Valid state config - must reference projections that have Tags
 */
type ValidStateConfig<TProjections extends readonly ProjectionComponent[]> =
  Record<string, ProjectionsWithTags<TProjections>>

/**
 * Expose config constrained to what projections actually provide
 */
export interface ExposeConfig<TProjections extends readonly ProjectionComponent[] = readonly ProjectionComponent[]> {
  readonly signals?: ValidSignalsConfig<TProjections>
  readonly state?: ValidStateConfig<TProjections>
}

// ---------------------------------------------------------------------------
// Client Types - extract actual types from the config
// ---------------------------------------------------------------------------

/** Extract the value type from a Signal<T> */
type SignalValue<T> = T extends Signal<infer V> ? V : never

/**
 * Extract state type from a regular projection's Tag (has .get)
 */
type ProjectionStateFromTag<T> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends { readonly Tag: Context.Tag<infer Service, any> }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ? Service extends { readonly get: Effect.Effect<infer S, any, any> }
      ? S
      : never
    : never

/**
 * Extract fork state type from a forked projection's Tag (has .getFork)
 */
type ForkedProjectionStateFromTag<T> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends { readonly Tag: Context.Tag<infer Service, any> }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ? Service extends { readonly getFork: (forkId: string | null) => Effect.Effect<infer S, any, any> }
      ? S
      : never
    : never

/**
 * Check if a projection is forked (has getFork method)
 */
type IsForkedProjection<T> =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  T extends { readonly Tag: Context.Tag<infer Service, any> }
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ? Service extends { readonly getFork: (forkId: string | null) => Effect.Effect<any, any, any> }
      ? true
      : false
    : false

/** Client signal subscription API */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ClientSignals<T extends Record<string, Signal<any>>> = {
  [K in keyof T]: (callback: (value: SignalValue<T[K]>) => void) => () => void
}

/** Client state API for regular projections */
type RegularClientState<T> = {
  get: () => Promise<ProjectionStateFromTag<T>>
  subscribe: (callback: (state: ProjectionStateFromTag<T>) => void) => () => void
}

/** Client state API for forked projections */
type ForkedClientState<T> = {
  getFork: (forkId: string | null) => Promise<ForkedProjectionStateFromTag<T>>
  subscribeFork: (forkId: string | null, callback: (state: ForkedProjectionStateFromTag<T>) => void) => () => void
}

/** Client state subscription API - different interface for regular vs forked */
type ClientState<T extends Record<string, ProjectionWithTag>> = {
  [K in keyof T]: IsForkedProjection<T[K]> extends true
    ? ForkedClientState<T[K]>
    : RegularClientState<T[K]>
}

/** The client interface returned by createClient */
export interface AgentClient<
  TEvent extends BaseEvent,
  TExpose extends ExposeConfig
> {
  /** Subscribe to exposed signals */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  readonly on: TExpose['signals'] extends Record<string, Signal<any>>
    ? ClientSignals<TExpose['signals']>
    : Record<string, never>

  /** Access exposed state */
  readonly state: TExpose['state'] extends Record<string, ProjectionWithTag>
    ? ClientState<TExpose['state']>
    : Record<string, never>

  /** Send an event to the agent */
  readonly send: (event: TEvent) => Promise<void>

  /** Subscribe to all events flowing through the bus */
  readonly onEvent: (callback: (event: TEvent) => void) => () => void

  /** Interrupt the agent - stops streaming and resets state */
  readonly interrupt: () => Promise<void>

  /**
   * Run an Effect within the agent's managed runtime.
   * Provides access to all internal services (projections, workers, core services).
   */
  readonly runEffect: <A, E, R>(effect: Effect.Effect<A, E, R>) => Promise<A>

  /** Subscribe to framework errors (handler failures, sink errors, etc.) */
  readonly onError: (callback: (error: FrameworkError) => void) => () => void

  /** Dispose the client and cleanup resources */
  readonly dispose: () => Promise<void>
}

// ---------------------------------------------------------------------------
// Agent Result
// ---------------------------------------------------------------------------

export interface AgentResult<
  TEvent extends BaseEvent,
  TProjectionOutputs,
  TProjections extends readonly ProjectionComponent[],
  TExpose extends ExposeConfig<TProjections>,
  TWorkerRequirements = never
> {
  readonly Layer: Layer.Layer<CoreServices<TEvent> | TProjectionOutputs, never, TWorkerRequirements>
  readonly expose: TExpose
  /** Projections registered with this agent (for tooling/visualization) */
  readonly projections: TProjections
  /**
   * Create an agent client.
   * If workers have external requirements, you must provide a layer that satisfies them.
   */
  readonly createClient: [TWorkerRequirements] extends [never]
    ? () => Promise<AgentClient<TEvent, TExpose>>
    : (requirements: Layer.Layer<TWorkerRequirements, never, never>) => Promise<AgentClient<TEvent, TExpose>>
}

// ---------------------------------------------------------------------------
// Agent.define()
// ---------------------------------------------------------------------------

export function define<TEvent extends BaseEvent>() {
  return <
    const TProjections extends readonly ProjectionComponent[],
    const TWorkers extends readonly WorkerComponent[],
    const TExpose extends ExposeConfig<TProjections> = Record<string, never>
  >(config: {
    name: string
    projections: TProjections
    workers: TWorkers
    expose?: TExpose
  }): AgentResult<
    TEvent,
    ExtractProjectionOutputs<TProjections>,
    TProjections,
    TExpose,
    ExtractExternalRequirements<TEvent, TProjections, TWorkers>
  > => {
    type TAllServices = CoreServices<TEvent> | ExtractProjectionOutputs<TProjections>

    const ProjectionBusLayer = makeProjectionBusLayer<TEvent>()
    const EventBusCoreLayer = makeEventBusCoreLayer<TEvent>()
    const WorkerBusLayer = makeWorkerBusLayer<TEvent>()

    // FrameworkErrorReporterLive depends on FrameworkErrorPubSubLive — must be explicitly wired
    // (Layer.mergeAll only merges outputs, it does NOT provide one layer's output as another's input at runtime)
    const FrameworkErrorReporterProvided = Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive)
    const CoreDeps = Layer.mergeAll(HydrationContext.Default, makeEventSinkLayer<TEvent>(), InterruptCoordinatorLive, FrameworkErrorPubSubLive, FrameworkErrorReporterProvided)
    const WithProjectionBus = Layer.provideMerge(ProjectionBusLayer, CoreDeps)
    const WithEventBusCore = Layer.provideMerge(EventBusCoreLayer, WithProjectionBus)
    const WithWorkerBus = Layer.provideMerge(WorkerBusLayer, WithEventBusCore)

    const BaseLayer = WithWorkerBus

    // Merge all projection layers
    const projectionLayers = config.projections.map(p => p.Layer) as Layer.Layer<unknown, never, unknown>[]
    const ProjectionsLayer = projectionLayers.length > 0
      ? projectionLayers.reduce((acc, l) => Layer.provideMerge(l, acc))
      : Layer.empty

    // Merge all worker layers
    const workerLayers = config.workers.map(w => w.Layer) as Layer.Layer<unknown, never, unknown>[]
    const WorkersLayer = workerLayers.length > 0
      ? workerLayers.reduce((acc, l) => Layer.provideMerge(l, acc))
      : Layer.empty

    // Compose: BaseLayer provides core services, ProjectionsLayer provides signals to workers
    // External requirements flow through (not cast away)
    type TExternalReqs = ExtractExternalRequirements<TEvent, TProjections, TWorkers>
    const AppLayer = Layer.provideMerge(
      WorkersLayer,
      Layer.provideMerge(ProjectionsLayer, BaseLayer)
    ) as Layer.Layer<TAllServices, never, TExternalReqs>

    const expose = (config.expose ?? {}) as TExpose

    // createClient accepts optional requirements layer
    const createClient = async (
      requirementsLayer?: Layer.Layer<TExternalReqs, never, never>
    ): Promise<AgentClient<TEvent, TExpose>> => {
      // Compose requirements layer if provided (use empty layer if not)
      const ReqLayer = (requirementsLayer ?? Layer.empty) as Layer.Layer<TExternalReqs, never, never>
      const FinalLayer = Layer.provideMerge(AppLayer, ReqLayer)
      const runtime = ManagedRuntime.make(FinalLayer)
      const BusTag = WorkerBusTag<TEvent>()
      const ProjBusTag = ProjectionBusTag<TEvent>()

      // Validate no cycles in dependency graph after all projections are registered
      await runtime.runPromise(
        Effect.gen(function* () {
          const bus = yield* ProjBusTag
          yield* bus.validateNoCycles()
        }) as Effect.Effect<void, never, TAllServices>
      )

      // Track active subscriptions for cleanup
      const activeSubscriptions: Array<() => void> = []

      // Helper: run a subscription effect with error reporting
      const runSubscription = (subscriptionName: string, effect: Effect.Effect<void, never, TAllServices>) => {
        const guarded = effect.pipe(
          Effect.catchAllCause((cause) =>
            Effect.flatMap(FrameworkErrorReporter, (reporter) =>
              reporter.report(FrameworkError.SubscriptionError({ subscriptionName, cause }))
            )
          )
        ) as Effect.Effect<void, never, TAllServices>
        runtime.runPromise(guarded)
      }

      // Build signal subscription handlers
      const onHandlers: Record<string, (callback: (value: unknown) => void) => () => void> = {}

      if (expose.signals) {
        for (const name of Object.keys(expose.signals)) {
          // Type assertion is safe: ValidSignalsConfig constrains this to Signal<T> from projections
          const signal = expose.signals[name] as Signal<unknown>
          onHandlers[name] = (callback) => {
            let isActive = true

            const effect = Effect.gen(function* () {
              const pubsub = yield* signal.tag
              yield* Stream.fromPubSub(pubsub).pipe(
                Stream.takeWhile(() => isActive),
                Stream.runForEach((value) =>
                  Effect.sync(() => {
                    if (isActive) callback(value)
                  })
                )
              )
            }) as Effect.Effect<void, never, TAllServices>

            runSubscription(`signal:${name}`, effect)

            const unsubscribe = () => { isActive = false }
            activeSubscriptions.push(unsubscribe)
            return unsubscribe
          }
        }
      }

      // Build state access handlers
      // Supports both regular projections (get/subscribe) and forked projections (getFork/subscribeFork)
      const stateHandlers: Record<string, unknown> = {}

      if (expose.state) {
        for (const [name, projection] of Object.entries(expose.state)) {
          const proj = projection as ProjectionWithTag
          const projTag = proj.Tag

          if (proj.isForked) {
            // Forked projection - provide getFork/subscribeFork
            stateHandlers[name] = {
              getFork: (forkId: string | null) => {
                const effect = Effect.gen(function* () {
                  const p = yield* projTag
                  return yield* p.getFork(forkId)
                }) as Effect.Effect<unknown, never, TAllServices>

                return runtime.runPromise(effect)
              },

              subscribeFork: (forkId: string | null, callback: (state: unknown) => void) => {
                let isActive = true

                const effect = Effect.gen(function* () {
                  const p = yield* projTag

                  // Emit initial state for this fork
                  const initialForkState = yield* p.getFork(forkId)
                  if (isActive) callback(initialForkState)

                  // Subscribe to changes via getFork
                  const stateRef = p.state
                  yield* stateRef.changes.pipe(
                    Stream.takeWhile(() => isActive),
                    Stream.mapEffect(() => p.getFork(forkId)),
                    Stream.changes,  // Only emit when fork state actually changes
                    Stream.runForEach((forkState) =>
                      Effect.sync(() => {
                        if (isActive) callback(forkState)
                      })
                    )
                  )
                }) as Effect.Effect<void, never, TAllServices>

                runSubscription(`state:${name}:fork`, effect)

                const unsubscribe = () => { isActive = false }
                activeSubscriptions.push(unsubscribe)
                return unsubscribe
              }
            }
          } else {
            // Standard projection - provide get/subscribe
            stateHandlers[name] = {
              get: () => {
                const effect = Effect.gen(function* () {
                  const p = yield* projTag
                  return yield* p.get
                }) as Effect.Effect<unknown, never, TAllServices>

                return runtime.runPromise(effect)
              },

              subscribe: (callback: (state: unknown) => void) => {
                let isActive = true

                const effect = Effect.gen(function* () {
                  const p = yield* projTag
                  const stateRef = p.state

                  // Emit initial state
                  const initial = yield* SubscriptionRef.get(stateRef)
                  if (isActive) callback(initial)

                  // Subscribe to changes
                  yield* stateRef.changes.pipe(
                    Stream.takeWhile(() => isActive),
                    Stream.runForEach((state) =>
                      Effect.sync(() => {
                        if (isActive) callback(state)
                      })
                    )
                  )
                }) as Effect.Effect<void, never, TAllServices>

                runSubscription(`state:${name}`, effect)

                const unsubscribe = () => { isActive = false }
                activeSubscriptions.push(unsubscribe)
                return unsubscribe
              }
            }
          }
        }
      }

      return {
        on: onHandlers as AgentClient<TEvent, TExpose>['on'],
        state: stateHandlers as AgentClient<TEvent, TExpose>['state'],

        send: (event: TEvent) => runtime.runPromise(
          Effect.flatMap(BusTag, (bus) => bus.publish(event))
        ),

        onEvent: (callback: (event: TEvent) => void) => {
          let isActive = true

          const effect = Effect.gen(function* () {
            const bus = yield* BusTag
            yield* bus.stream.pipe(
              Stream.takeWhile(() => isActive),
              Stream.runForEach((event) =>
                Effect.sync(() => {
                  if (isActive) callback(event)
                })
              )
            )
          }) as Effect.Effect<void, never, TAllServices>

          runSubscription('onEvent', effect)

          const unsubscribe = () => { isActive = false }
          activeSubscriptions.push(unsubscribe)
          return unsubscribe
        },

        runEffect: <A, E, R>(effect: Effect.Effect<A, E, R>) =>
          runtime.runPromise(effect as Effect.Effect<A, E, TAllServices>),

        onError: (callback: (error: FrameworkError) => void) => {
          let isActive = true

          const effect = Effect.gen(function* () {
            const pubsub = yield* FrameworkErrorPubSub
            yield* Stream.fromPubSub(pubsub).pipe(
              Stream.takeWhile(() => isActive),
              Stream.runForEach((error) =>
                Effect.sync(() => { if (isActive) callback(error) })
              )
            )
          }) as Effect.Effect<void, never, TAllServices>

          // onError itself can't use runSubscription (circular) — just swallow
          runtime.runPromise(effect)

          const unsubscribe = () => { isActive = false }
          activeSubscriptions.push(unsubscribe)
          return unsubscribe
        },

        interrupt: () => runtime.runPromise(
          Effect.flatMap(BusTag, (bus) => bus.publish({ type: 'interrupt' } as TEvent))
        ),

        dispose: async () => {
          for (const unsub of activeSubscriptions) {
            unsub()
          }
          await runtime.dispose()
        }
      }
    }

    return {
      Layer: AppLayer,
      expose,
      projections: config.projections,
      createClient
    }
  }
}
