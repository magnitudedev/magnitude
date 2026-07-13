import { Effect, Layer, Stream, SubscriptionRef, PubSub, Context, Fiber, Schema } from 'effect'
import { EventBusCoreTag, makeEventBusCoreLayer, type BaseEvent, type EventBusCoreService } from '../core/event-bus-core'
import { EventSinkTag, makeEventSinkLayer, type EventSinkService } from '../core/event-sink'
import { HydrationContext } from '../core/hydration-context'
import { InterruptCoordinatorLive } from '../core/interrupt-coordinator'
import { makeProjectionBusLayer, ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
import { makeAmbientServiceLayer, AmbientServiceTag, type AmbientService } from '../core/ambient-service'
import { makeWorkerBusLayer, WorkerBusTag, type WorkerBusService } from '../core/worker-bus'
import { consumer as ProjectionConsumer } from '../projection'
import { type Signal } from '../signal/define'
import type { AmbientDef, AmbientRequirementsOf } from '../ambient/define'
import type { AddressedEntryStore } from '../addressed/entry-store'
import type { AddressedError } from '../addressed/errors'
import { EventCursorSchema, type EventCursor } from '../core/event-cursor'
import type { ProjectionInstance, ProjectionResult } from '../projection/define'
import type { ForkedProjectionInstance, ForkedProjectionResult, ForkedProjectionSnapshot } from '../projection/defineForked'
import type { ParseResult } from 'effect'
import {
  FrameworkErrorPubSub,
  FrameworkErrorPubSubLive,
  FrameworkErrorReporter,
  FrameworkErrorReporterLive,
  FrameworkError,
  type FrameworkErrorReporterService
} from '../core/framework-error'
import { createManagedClient, type Client } from './client'
import {
  ProjectionSnapshotEnvelopeInvalid,
  ProjectionSnapshotProjectionInvalid,
  ProjectionSnapshotProjectionSetMismatch,
  ProjectionSnapshotServiceTag,
  type ProjectionSnapshotInvalid,
  type ProjectionSnapshotRestorePlan,
  type ProjectionSnapshotService
} from './projection-snapshot-service'
import {
  RuntimeIntrospector,
  makeRuntimeIntrospectionService,
  type RuntimeProjectionInspector
} from '../introspection/runtime'

// ---------------------------------------------------------------------------
// EventEngine Service
// ---------------------------------------------------------------------------

/**
 * The EventEngine is the Effect-native core of the agent system.
 * It is a Layer.scoped service that owns all subscription fibers via forkIn.
 * All long-lived fibers are tied to the engine's scope, which is closed by
 * ManagedRuntime.dispose() — auto-interrupting all fibers.
 */
export interface Shape<
  TEvent extends BaseEvent,
  TExpose,
  TProjections extends readonly ProjectionComponent[] = readonly ProjectionComponent[]
> {
  // One-shot operations
  send(event: TEvent): Effect.Effect<void>
  interrupt(): Effect.Effect<void>
  stateGet(name: string): Effect.Effect<unknown>
  stateGetFork(name: string, forkId: string | null): Effect.Effect<unknown>
  captureProjectionSnapshot(cursor: EventCursor, sessionId: string): Effect.Effect<ProjectionSnapshotEnvelope<TProjections>, ParseResult.ParseError | AddressedError>
  prepareProjectionSnapshotRestore(snapshot: unknown): Effect.Effect<ProjectionSnapshotRestorePlan, ProjectionSnapshotInvalid>

  // Effect-native observation
  readonly events: Stream.Stream<TEvent>
  readonly errors: Stream.Stream<FrameworkError>

  // Subscriptions — each creates a forkIn fiber and returns a Fiber handle
  subscribeSignal(name: string, callback: (value: unknown) => void): Effect.Effect<Fiber.RuntimeFiber<void, never>>
  subscribeState(name: string, callback: (state: unknown) => void): Effect.Effect<Fiber.RuntimeFiber<void, never>>
  subscribeStateFork(name: string, forkId: string | null, callback: (state: unknown) => void): Effect.Effect<Fiber.RuntimeFiber<void, never>>
  subscribeEvent(callback: (event: TEvent) => void): Effect.Effect<Fiber.RuntimeFiber<void, never>>
  subscribeError(callback: (error: FrameworkError) => void): Effect.Effect<Fiber.RuntimeFiber<void, never>>
}

export const Service = Context.GenericTag<Shape<BaseEvent, unknown>>('EventEngine')

// ---------------------------------------------------------------------------
// Core Services
// ---------------------------------------------------------------------------

export type CoreServices<TEvent extends BaseEvent> =
  | HydrationContext
  | EventSinkService<TEvent>
  | EventBusCoreService<TEvent>
  | ProjectionBusService<TEvent>
  | AmbientService
  | WorkerBusService<TEvent>
  | ProjectionConsumer.ProjectionConsumerService
  | ProjectionSnapshotService
  | RuntimeIntrospector
  | FrameworkErrorReporterService
  | PubSub.PubSub<FrameworkError>

// ---------------------------------------------------------------------------
// Component Interfaces
// ---------------------------------------------------------------------------

type ProjectionComponent =
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ProjectionResult<string, Schema.Schema.AnyNoContext, any, any, any, any>
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  | ForkedProjectionResult<string, Schema.Schema.AnyNoContext, any, any, any, any>

interface WorkerComponent {
  readonly Layer: Layer.Layer<never, never, unknown>
}

/** Extract ambient requirements from a projection's declared ambients */
type ExtractProjectionAmbientRequirements<P> =
  P extends { readonly ambients?: infer TAmbients }
    ? TAmbients extends readonly unknown[]
      ? AmbientRequirementsOf<TAmbients[number]>
      : never
    : never

/** Extract all ambient requirements as a union */
type ExtractAllProjectionAmbientRequirements<T extends readonly ProjectionComponent[]> =
  ExtractProjectionAmbientRequirements<T[number]>

/** Extract addressed store requirements from addressed projections. */
type ExtractProjectionAddressedRequirements<P> =
  P extends ProjectionResult<any, any, any, any, any, infer TAddressed>
    ? keyof TAddressed extends never ? never : AddressedEntryStore
    : P extends ForkedProjectionResult<any, any, any, any, any, infer TAddressed>
      ? keyof TAddressed extends never ? never : AddressedEntryStore
    : never

/** Extract all addressed store requirements from projections. */
type ExtractAllProjectionAddressedRequirements<T extends readonly ProjectionComponent[]> =
  ExtractProjectionAddressedRequirements<T[number]>

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
> =
  | Exclude<
    ExtractAllWorkerRequirements<TWorkers> | ExtractAllProjectionAddressedRequirements<TProjections>,
    InternalRequirements<TEvent, TProjections>
  >
  | ExtractAllProjectionAmbientRequirements<TProjections>

type ExtractLayerOutput<L> = L extends Layer.Layer<infer Out, infer _E, infer _R> ? Out : never
type ExtractProjectionOutputs<T extends readonly ProjectionComponent[]> =
  ExtractLayerOutput<T[number]['Layer']>

// ---------------------------------------------------------------------------
// Type Extractors for Projections
// ---------------------------------------------------------------------------

type ProjectionSnapshotValue<P> =
  P extends ProjectionResult<any, infer StateSchema, any, any, any, any>
    ? Schema.Schema.Encoded<StateSchema>
    : P extends ForkedProjectionResult<any, infer ForkStateSchema, any, any, any, any>
      ? ForkedProjectionSnapshot<ForkStateSchema>
      : never

export type ProjectionSnapshotRecord<TProjections extends readonly ProjectionComponent[]> =
  Readonly<Record<string, ProjectionSnapshotValue<TProjections[number]>>>

export interface ProjectionSnapshotEnvelope<TProjections extends readonly ProjectionComponent[] = readonly ProjectionComponent[]> {
  readonly sessionId?: string
  readonly engineName?: string
  /** Ignored legacy metadata. Snapshot compatibility is validated from projection payloads. */
  readonly schemaVersion?: string
  readonly eventCursor: EventCursor
  readonly projections: ProjectionSnapshotRecord<TProjections>
}

type ProjectionInstanceFor<P> =
  P extends ProjectionResult<any, infer StateSchema, any, any, any, any>
    ? ProjectionInstance<StateSchema>
    : P extends ForkedProjectionResult<any, infer ForkStateSchema, any, any, any, any>
      ? ForkedProjectionInstance<ForkStateSchema>
      : never

const makeProjectionSnapshotEnvelopeSchema = <TProjections extends readonly ProjectionComponent[]>(): Schema.Schema<ProjectionSnapshotEnvelope<TProjections>> => {
  const projectionRecordSchema = Schema.Record({
    key: Schema.String,
    value: Schema.Unknown,
  }) as Schema.Schema<ProjectionSnapshotRecord<TProjections>>

  return Schema.Struct({
    sessionId: Schema.optional(Schema.String),
    engineName: Schema.optional(Schema.String),
    schemaVersion: Schema.optional(Schema.String),
    eventCursor: EventCursorSchema,
    projections: projectionRecordSchema,
  })
}

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
  T[number]

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
// EventEngine Result
// ---------------------------------------------------------------------------

export interface Result<
  TEvent extends BaseEvent,
  TProjectionOutputs,
  TProjections extends readonly ProjectionComponent[],
  TExpose extends ExposeConfig<TProjections>,
  TWorkerRequirements = never
> {
  readonly Layer: Layer.Layer<CoreServices<TEvent> | TProjectionOutputs, AddressedError, TWorkerRequirements>
  readonly EngineLayer: Layer.Layer<
    CoreServices<TEvent> | TProjectionOutputs | Shape<TEvent, TExpose, TProjections>,
    AddressedError,
    TWorkerRequirements
  >
  readonly expose: TExpose
  /** Projections registered with this agent (for tooling/visualization) */
  readonly projections: TProjections
  /**
   * Create a managed Promise client.
   * If workers have external requirements, you must provide a layer that satisfies them.
   */
  readonly createClient: [TWorkerRequirements] extends [never]
    ? () => Promise<Client<TEvent, TExpose, CoreServices<TEvent> | TProjectionOutputs>>
    : (requirements: Layer.Layer<TWorkerRequirements, never, never>) => Promise<Client<TEvent, TExpose, CoreServices<TEvent> | TProjectionOutputs>>
}

// ---------------------------------------------------------------------------
// EventEngine.make()
// ---------------------------------------------------------------------------

export function make<TEvent extends BaseEvent>() {
  return <
    const TProjections extends readonly ProjectionComponent[],
    const TWorkers extends readonly WorkerComponent[],
    const TExpose extends ExposeConfig<TProjections> = Record<string, never>
  >(config: {
    name: string
    schemaVersion: string
    projections: TProjections
    workers: TWorkers
    expose?: TExpose
  }): Result<
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
    const AmbientServiceLayer = makeAmbientServiceLayer<TEvent>()
    const ProjectionConsumerLayer = ProjectionConsumer.Live

    // FrameworkErrorReporterLive depends on FrameworkErrorPubSubLive — must be explicitly wired
    // (Layer.mergeAll only merges outputs, it does NOT provide one layer's output as another's input at runtime)
    const FrameworkErrorReporterProvided = Layer.provide(FrameworkErrorReporterLive, FrameworkErrorPubSubLive)
    const CoreDeps = Layer.mergeAll(HydrationContext.Default, makeEventSinkLayer<TEvent>(), InterruptCoordinatorLive, FrameworkErrorPubSubLive, FrameworkErrorReporterProvided)
    const WithProjectionBus = Layer.provideMerge(ProjectionBusLayer, CoreDeps)
    const WithAmbientService = Layer.provideMerge(AmbientServiceLayer, WithProjectionBus)
    const WithEventBusCore = Layer.provideMerge(EventBusCoreLayer, WithAmbientService)
    const WithWorkerBus = Layer.provideMerge(WorkerBusLayer, WithEventBusCore)
    const WithProjectionConsumer = Layer.provideMerge(ProjectionConsumerLayer, WithWorkerBus)

    const BaseLayer = WithProjectionConsumer

    // Merge all projection layers
    const projectionLayers = config.projections.map(p => p.Layer) as Layer.Layer<unknown, AddressedError, unknown>[]
    const ProjectionsLayer = projectionLayers.length > 0
      ? projectionLayers.reduce((acc, l) => Layer.provideMerge(l, acc))
      : Layer.empty

    // Merge all worker layers
    const workerLayers = config.workers.map(w => w.Layer) as Layer.Layer<unknown, never, unknown>[]
    const WorkersLayer = workerLayers.length > 0
      ? workerLayers.reduce((acc, l) => Layer.provideMerge(l, acc))
      : Layer.empty

    type TExternalReqs = ExtractExternalRequirements<TEvent, TProjections, TWorkers>
    const expose = (config.expose ?? {}) as TExpose
    const projectionSnapshotEnvelopeSchema = makeProjectionSnapshotEnvelopeSchema<TProjections>()

    const ProjectionSnapshotServiceLayer = Layer.scoped(
      ProjectionSnapshotServiceTag,
      Effect.gen(function* () {
        const snapshotServices = new Map<string, ProjectionInstanceFor<ProjectionComponent>>()
        for (const projection of config.projections) {
          snapshotServices.set(
            projection.name,
            yield* projection.Tag
          )
        }

        const captureProjectionSnapshot = (cursor: EventCursor, sessionId: string) =>
          Effect.gen(function* () {
            const projections: Record<string, ProjectionSnapshotValue<TProjections[number]>> = {}
            for (const [name, projection] of snapshotServices) {
              projections[name] = yield* projection.snapshot
            }
            return {
              sessionId,
              engineName: config.name,
              eventCursor: cursor,
              projections,
            }
          })

        const prepareProjectionSnapshotRestore = (snapshot: unknown) => Effect.gen(function* () {
          const envelope = yield* Schema.decodeUnknown(projectionSnapshotEnvelopeSchema)(snapshot).pipe(
            Effect.mapError((cause) => new ProjectionSnapshotEnvelopeInvalid({ cause }))
          )
          const projectionSnapshots = envelope.projections
          const expectedNames = new Set(snapshotServices.keys())
          const actualNames = new Set(Object.keys(projectionSnapshots))
          const missing = Array.from(expectedNames).filter((name) => !actualNames.has(name))
          const extra = Array.from(actualNames).filter((name) => !expectedNames.has(name))

          if (missing.length > 0 || extra.length > 0) {
            return yield* new ProjectionSnapshotProjectionSetMismatch({ missing, extra })
          }

          const restorePlans: Effect.Effect<void>[] = []
          for (const [name, projection] of snapshotServices) {
            const plan = yield* projection.prepareRestore(projectionSnapshots[name]).pipe(
              Effect.mapError((cause) => new ProjectionSnapshotProjectionInvalid({
                projectionName: name,
                cause,
              }))
            )
            restorePlans.push(plan.commit)
          }

          const commit = Effect.gen(function* () {
            for (const restoreCommit of restorePlans) {
              yield* restoreCommit
            }
          }).pipe(Effect.uninterruptible)

          return {
            eventCursor: envelope.eventCursor,
            commit,
          } satisfies ProjectionSnapshotRestorePlan
        })

        return {
          captureProjectionSnapshot,
          prepareProjectionSnapshotRestore,
        } satisfies ProjectionSnapshotService
      })
    )

    const RuntimeIntrospectionLayer = Layer.scoped(
      RuntimeIntrospector,
      Effect.gen(function* () {
        const inspectors: RuntimeProjectionInspector[] = []

        for (const projection of config.projections) {
          const service = yield* projection.Tag
          const signalSubscriptions = projection.signalSubscriptions.map((subscription) => subscription.signal)

          if (projection.isForked) {
            const forked = service as ForkedProjectionInstance<Schema.Schema.AnyNoContext>
            inspectors.push({
              name: projection.name,
              kind: 'forked',
              reads: projection.reads,
              signalSubscriptions,
              read: (forkId) => forked.getFork(forkId),
              changes: forked.state.changes.pipe(Stream.map(() => undefined)),
            })
          } else {
            const global = service as ProjectionInstance<Schema.Schema.AnyNoContext>
            inspectors.push({
              name: projection.name,
              kind: 'global',
              reads: projection.reads,
              signalSubscriptions,
              read: () => global.get,
              changes: global.state.changes.pipe(Stream.map(() => undefined)),
            })
          }
        }

        return makeRuntimeIntrospectionService(
          config.name,
          config.schemaVersion,
          inspectors
        )
      })
    )

    const ProjectionAppLayer = Layer.provideMerge(
      Layer.mergeAll(ProjectionSnapshotServiceLayer, RuntimeIntrospectionLayer),
      Layer.provideMerge(ProjectionsLayer, BaseLayer)
    )

    // Compose: BaseLayer provides core services, ProjectionsLayer provides
    // signals/state, ProjectionSnapshotService provides snapshot capture/restore,
    // and workers sit on top.
    const AppLayer = Layer.provideMerge(
      WorkersLayer,
      ProjectionAppLayer
    ) as Layer.Layer<TAllServices, AddressedError, TExternalReqs>

    // ---------------------------------------------------------------------------
    // EventEngineLive - scoped service that owns all subscription fibers
    // ---------------------------------------------------------------------------
    const EventEngineLive = Layer.scoped(
      Service,
      Effect.gen(function* () {
        const engineScope = yield* Effect.scopeWith((scope) => Effect.succeed(scope))

        const BusTag = WorkerBusTag<TEvent>()
        const ProjBusTag = ProjectionBusTag<TEvent>()

        const bus = yield* BusTag
        const projectionBus = yield* ProjBusTag
        yield* projectionBus.validateNoCycles()
        const frameworkErrorPubSub = yield* FrameworkErrorPubSub
        const frameworkErrorReporter = yield* FrameworkErrorReporter
        const projectionSnapshotService = yield* ProjectionSnapshotServiceTag

        const signalPubSubs = new Map<string, PubSub.PubSub<unknown>>()
        if (expose.signals) {
          for (const [name, signal] of Object.entries(expose.signals)) {
            signalPubSubs.set(name, yield* (signal as Signal<unknown>).tag)
          }
        }

        type ExposedProjectionService = {
          readonly get?: Effect.Effect<unknown>
          readonly getFork?: (forkId: string | null) => Effect.Effect<unknown>
          readonly state: SubscriptionRef.SubscriptionRef<unknown>
        }
        const stateServices = new Map<string, ExposedProjectionService>()
        if (expose.state) {
          for (const [name, projection] of Object.entries(expose.state)) {
            stateServices.set(name, (yield* projection.Tag) as ExposedProjectionService)
          }
        }

        const guardAndFork = (
          name: string,
          effect: Effect.Effect<void>
        ) =>
          Effect.forkIn(
            effect.pipe(
              Effect.catchAllCause((cause) =>
                frameworkErrorReporter.report(
                  FrameworkError.SubscriptionError({ subscriptionName: name, cause })
                )
              )
            ),
            engineScope
          )

        const engine = {
          send: (event: TEvent) => bus.publish(event),

          interrupt: () => bus.publish({ type: 'interrupt' } as TEvent),

          events: bus.stream,

          errors: Stream.fromPubSub(frameworkErrorPubSub),

          stateGet: (name: string) => Effect.gen(function* () {
            const projection = stateServices.get(name)
            if (!projection?.get) return undefined
            return yield* projection.get
          }),

          stateGetFork: (name: string, forkId: string | null) => Effect.gen(function* () {
            const projection = stateServices.get(name)
            if (!projection?.getFork) return undefined
            return yield* projection.getFork(forkId)
          }),

          captureProjectionSnapshot: (cursor: EventCursor, sessionId: string) =>
            projectionSnapshotService.captureProjectionSnapshot(cursor, sessionId) as Effect.Effect<ProjectionSnapshotEnvelope<TProjections>, ParseResult.ParseError | AddressedError>,

          prepareProjectionSnapshotRestore: (snapshot: unknown) =>
            projectionSnapshotService.prepareProjectionSnapshotRestore(snapshot),

          subscribeSignal: (name: string, callback: (value: unknown) => void) => Effect.gen(function* () {
            const pubsub = signalPubSubs.get(name)
            if (!pubsub) return yield* Effect.die(new Error(`Unknown signal: ${name}`))
            return yield* guardAndFork(
              `signal:${name}`,
              Stream.runForEach(Stream.fromPubSub(pubsub), (value) =>
                Effect.sync(() => callback(value))
              )
            )
          }),

          subscribeState: (name: string, callback: (state: unknown) => void) => Effect.gen(function* () {
            const p = stateServices.get(name)
            if (!p) return yield* Effect.die(new Error(`Unknown state: ${name}`))
            const initial = yield* SubscriptionRef.get(p.state)
            yield* Effect.sync(() => callback(initial))
            return yield* guardAndFork(
              `state:${name}`,
              Stream.runForEach(p.state.changes, (state) =>
                Effect.sync(() => callback(state))
              )
            )
          }),

          subscribeStateFork: (name: string, forkId: string | null, callback: (state: unknown) => void) => Effect.gen(function* () {
            const p = stateServices.get(name)
            if (!p?.getFork) return yield* Effect.die(new Error(`Unknown forked state: ${name}`))
            const getFork = p.getFork
            const initial = yield* getFork(forkId)
            yield* Effect.sync(() => callback(initial))
            return yield* guardAndFork(
              `state:${name}:fork`,
              Stream.runForEach(
                p.state.changes.pipe(
                  Stream.mapEffect(() => getFork(forkId)),
                  Stream.changes
                ),
                (forkState) => Effect.sync(() => callback(forkState))
              )
            )
          }),

          subscribeEvent: (callback: (event: TEvent) => void) =>
            guardAndFork(
              'onEvent',
              Stream.runForEach(bus.stream, (event) =>
                Effect.sync(() => callback(event))
              )
            ),

          subscribeError: (callback: (error: FrameworkError) => void) =>
            guardAndFork(
              'onError',
              Stream.runForEach(Stream.fromPubSub(frameworkErrorPubSub), (error) =>
                Effect.sync(() => callback(error))
              )
            )
        } satisfies Shape<TEvent, TExpose, TProjections>

        return engine
      })
    )

    const EngineLayer = Layer.provideMerge(EventEngineLive, AppLayer) as Layer.Layer<
      TAllServices | Shape<TEvent, TExpose, TProjections>,
      AddressedError,
      TExternalReqs
    >

    const createClient = async (
      requirementsLayer?: Layer.Layer<TExternalReqs, never, never>
    ): Promise<Client<TEvent, TExpose, TAllServices>> =>
      createManagedClient<TEvent, TExpose, TAllServices, TExternalReqs>({
        engineLayer: EngineLayer,
        requirementsLayer,
        expose,
        getEngine: (context) =>
          Context.unsafeGet(context, Service) as Shape<TEvent, TExpose, TProjections>
      })

    return {
      Layer: AppLayer,
      EngineLayer,
      expose,
      projections: config.projections,
      createClient
    }
  }
}
