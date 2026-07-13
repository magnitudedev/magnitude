import { Context, Effect, Layer, Ref, Scope, Stream } from 'effect'
import type { Schema } from 'effect'
import { ProjectionBusTag, type ProjectionBusService } from '../core/projection-bus'
import type { AddressedError } from '../addressed/errors'
import {
  makeReadTracker,
  type AddressedReadTracker,
  type ProjectionAddressedConsumers,
  type ProjectionForkedAddressedConsumers
} from './addressed'
import type { AnyProjectionResult, ProjectionResult } from './define'
import type { ForkedProjectionResult, ForkedState } from './defineForked'

const RuntimeConsumerInternal: unique symbol = Symbol('RuntimeConsumerInternal')

export interface RuntimeConsumer {
  readonly id: string
  readonly [RuntimeConsumerInternal]: {
    readonly observerName: string
    readonly changes: Stream.Stream<void>
  }
}

export interface ProjectionConsumerService {
  readonly nextObserverName: (id: string) => Effect.Effect<string>
}

export const Service = Context.GenericTag<ProjectionConsumerService>('ProjectionConsumer')

interface TrackingScope {
  readonly bus: ProjectionBusService<any>
  readonly observerName: string
  readonly projectionNames: Set<string>
  readonly addressedReads: AddressedReadTracker
}

const TrackingScope = Context.GenericTag<TrackingScope>('ProjectionConsumer/TrackingScope')

export type StateOf<P> =
  P extends ProjectionResult<string, infer StateSchema, any, any, any, any>
    ? Schema.Schema.Type<StateSchema>
    : P extends ForkedProjectionResult<string, infer ForkStateSchema, any, any, any, any>
      ? ForkedState<Schema.Schema.Type<ForkStateSchema>>
      : never

export type AddressedOf<P> =
  P extends ProjectionResult<string, any, any, any, any, infer TAddressed>
    ? ProjectionAddressedConsumers<TAddressed>
    : P extends ForkedProjectionResult<string, any, any, any, any, infer TAddressed>
      ? ProjectionForkedAddressedConsumers<TAddressed>
      : never

export interface ProjectionRead<P> {
  readonly state: StateOf<P>
  readonly addressed: AddressedOf<P>
}

export const Live: Layer.Layer<ProjectionConsumerService> =
  Layer.effect(
    Service,
    Effect.gen(function* () {
      const counter = yield* Ref.make(0)

      return {
        nextObserverName: (id) =>
          Ref.modify(counter, (next) => [
            `consumer:${id}:${next}`,
            next + 1
          ])
      } satisfies ProjectionConsumerService
    })
  )

export const acquire = (id: string): Effect.Effect<RuntimeConsumer, never, Scope.Scope | ProjectionConsumerService | ProjectionBusService<any>> =>
  Effect.acquireRelease(
    Effect.gen(function* () {
      const service = yield* Service
      const bus = yield* ProjectionBusTag<any>()
      const observerName = yield* service.nextObserverName(id)
      const changes = yield* bus.registerRuntimeObserver(observerName)

      return {
        id,
        [RuntimeConsumerInternal]: {
          observerName,
          changes
        }
      } satisfies RuntimeConsumer
    }),
    (consumer) =>
      Effect.gen(function* () {
        const bus = yield* ProjectionBusTag<any>()
        yield* bus.releaseRuntimeObserver(consumer[RuntimeConsumerInternal].observerName)
      }).pipe(Effect.ignore)
  )

export const provide = (consumer: RuntimeConsumer) =>
  <A, E, R>(
    effect: Effect.Effect<A, E, R>
  ): Effect.Effect<A, E | AddressedError, Exclude<R, TrackingScope> | ProjectionBusService<any>> =>
    Effect.gen(function* () {
      const bus = yield* ProjectionBusTag<any>()
      const tracking: TrackingScope = {
        bus,
        observerName: consumer[RuntimeConsumerInternal].observerName,
        projectionNames: new Set(),
        addressedReads: makeReadTracker()
      }

      const value = yield* effect.pipe(
        Effect.provideService(TrackingScope, tracking)
      )
      yield* bus.updateRuntimeObserverDependencies(
        tracking.observerName,
        tracking.projectionNames,
        tracking.addressedReads
      )
      return value
    })

export const stream = (consumer: RuntimeConsumer) =>
  <A, E, R>(
    effect: Effect.Effect<A, E, R>
  ): Stream.Stream<A, E | AddressedError, Exclude<R, TrackingScope> | ProjectionBusService<any>> =>
    Stream.concat(
      Stream.fromEffect(provide(consumer)(effect)),
      consumer[RuntimeConsumerInternal].changes.pipe(
        Stream.mapEffect(() => provide(consumer)(effect))
      )
    )

export const read = <P extends AnyProjectionResult>(
  projection: P
): Effect.Effect<ProjectionRead<P>, never, TrackingScope> =>
  Effect.map(TrackingScope, (scope) => {
    scope.projectionNames.add(projection.name)
    return {
      state: scope.bus.getProjectionState(projection.name) as StateOf<P>,
      addressed: scope.bus.getProjectionAddressedConsumersWithTracker(
        projection.name,
        scope.addressedReads
      ) as AddressedOf<P>
    }
  })
