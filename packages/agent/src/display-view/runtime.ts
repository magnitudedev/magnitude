import { Context, Data, Effect, Exit, Fiber, Layer, PubSub, Scope, Stream, SynchronizedRef } from 'effect'
import { Addressed, AmbientServiceTag, Projection, ProjectionBusTag } from '@magnitudedev/event-core'
import {
  sameDisplayViewShape,
  type DisplayViewShape,
  type DisplayViewSnapshot,
} from '@magnitudedev/protocol'
import { buildDisplayViewSnapshot } from './snapshot'

export class DisplayViewNotFoundError extends Data.TaggedError('DisplayViewNotFoundError')<{
  readonly viewId: string
}> {}

export class DisplayViewRuntimeError extends Data.TaggedError('DisplayViewRuntimeError')<{
  readonly viewId: string
  readonly operation: 'setShape' | 'stream'
  readonly cause: Addressed.AddressedError
}> {}

export interface DisplayViewRuntimeService {
  readonly setShape: (viewId: string, shape: DisplayViewShape) => Effect.Effect<void, DisplayViewRuntimeError>
  readonly stream: (viewId: string) => Stream.Stream<DisplayViewSnapshot, DisplayViewNotFoundError | DisplayViewRuntimeError>
  readonly snapshot: (viewId: string) => Effect.Effect<DisplayViewSnapshot, DisplayViewNotFoundError | DisplayViewRuntimeError>
  readonly close: (viewId: string) => Effect.Effect<void>
}

export class DisplayViewRuntime extends Context.Tag('DisplayViewRuntime')<
  DisplayViewRuntime,
  DisplayViewRuntimeService
>() {}

interface RuntimeDisplayViewEntry {
  readonly requestedShape: DisplayViewShape
  readonly snapshot: DisplayViewSnapshot
  readonly failure: DisplayViewRuntimeError | null
  readonly consumer: Projection.consumer.RuntimeConsumer
  readonly scope: Scope.CloseableScope
  readonly pubsub: PubSub.PubSub<RuntimeDisplayViewUpdate>
  readonly fiber: Fiber.RuntimeFiber<void, never>
  readonly generation: number
}

type RuntimeDisplayViewUpdate =
  | { readonly _tag: 'snapshot'; readonly snapshot: DisplayViewSnapshot }
  | { readonly _tag: 'failure'; readonly error: DisplayViewRuntimeError }

const closeEntry = (entry: RuntimeDisplayViewEntry): Effect.Effect<void> =>
  Fiber.interrupt(entry.fiber).pipe(
    Effect.zipRight(Scope.close(entry.scope, Exit.void)),
    Effect.asVoid
  )

const displayViewRuntimeError = (
  viewId: string,
  operation: DisplayViewRuntimeError['operation']
) =>
  (cause: Addressed.AddressedError) =>
    new DisplayViewRuntimeError({ viewId, operation, cause })

const displayViewUpdateEffect = (
  update: RuntimeDisplayViewUpdate
): Effect.Effect<DisplayViewSnapshot, DisplayViewRuntimeError> =>
  update._tag === 'snapshot'
    ? Effect.succeed(update.snapshot)
    : Effect.fail(update.error)

export const DisplayViewRuntimeLive =
  Layer.scoped(
    DisplayViewRuntime,
    Effect.gen(function* () {
      const runtimeScope = yield* Effect.scopeWith((scope) => Effect.succeed(scope))
      const ambientService = yield* AmbientServiceTag
      const projectionConsumerService = yield* Projection.consumer.Service
      const projectionBus = yield* ProjectionBusTag<any>()
      const views = yield* SynchronizedRef.make<ReadonlyMap<string, RuntimeDisplayViewEntry>>(new Map())

      const provideRuntimeEffect = <A, E, R>(
        effect: Effect.Effect<A, E, R>
      ) =>
        effect.pipe(
          Effect.provideService(AmbientServiceTag, ambientService),
          Effect.provideService(Projection.consumer.Service, projectionConsumerService),
          Effect.provideService(ProjectionBusTag<any>(), projectionBus)
        )

      const provideRuntimeStream = <A, E, R>(
        stream: Stream.Stream<A, E, R>
      ) =>
        stream.pipe(
          Stream.provideService(AmbientServiceTag, ambientService),
          Stream.provideService(Projection.consumer.Service, projectionConsumerService),
          Stream.provideService(ProjectionBusTag<any>(), projectionBus)
        )

      const publishSnapshotIfCurrent = (
        viewId: string,
        generation: number,
        snapshot: DisplayViewSnapshot
      ) =>
        SynchronizedRef.updateEffect(
          views,
          (currentViews) => {
            const current = currentViews.get(viewId)
            if (!current || current.generation !== generation) {
              return Effect.succeed(currentViews)
            }

            const nextViews = new Map(currentViews)
            nextViews.set(viewId, { ...current, snapshot, failure: null })
            return PubSub.publish(current.pubsub, { _tag: 'snapshot', snapshot }).pipe(
              Effect.as(nextViews)
            )
          }
        )

      const publishFailureIfCurrent = (
        viewId: string,
        generation: number,
        error: DisplayViewRuntimeError
      ) =>
        SynchronizedRef.updateEffect(
          views,
          (currentViews) => {
            const current = currentViews.get(viewId)
            if (!current || current.generation !== generation) {
              return Effect.succeed(currentViews)
            }

            const nextViews = new Map(currentViews)
            nextViews.set(viewId, { ...current, failure: error })
            return PubSub.publish(current.pubsub, { _tag: 'failure', error }).pipe(
              Effect.as(nextViews)
            )
          }
        )

      const startStream = (
        viewId: string,
        consumer: Projection.consumer.RuntimeConsumer,
        shape: DisplayViewShape,
        generation: number
      ) =>
        Projection.consumer.stream(consumer)(buildDisplayViewSnapshot(shape)).pipe(
          provideRuntimeStream,
          Stream.mapError(displayViewRuntimeError(viewId, 'stream')),
          Stream.runForEach((snapshot) => publishSnapshotIfCurrent(viewId, generation, snapshot)),
          Effect.catchAll((error) => publishFailureIfCurrent(viewId, generation, error)),
          Effect.forkIn(runtimeScope)
        )

      const makeEntry = (
        viewId: string,
        shape: DisplayViewShape,
        pubsub: PubSub.PubSub<RuntimeDisplayViewUpdate>,
        generation: number
      ) =>
        Effect.gen(function* () {
          const scope = yield* Scope.make()
          return yield* Effect.gen(function* () {
            const consumer = yield* Projection.consumer.acquire(`display-view:${viewId}`).pipe(
              Scope.extend(scope),
              provideRuntimeEffect
            )
            const snapshot = yield* buildDisplayViewSnapshot(shape).pipe(
              Projection.consumer.provide(consumer),
              provideRuntimeEffect
            )
            const fiber = yield* startStream(viewId, consumer, shape, generation)

            return {
              requestedShape: shape,
              snapshot,
              failure: null,
              consumer,
              scope,
              pubsub,
              fiber,
              generation,
            } satisfies RuntimeDisplayViewEntry
          }).pipe(
            Effect.tapError(() => Scope.close(scope, Exit.void)),
            Effect.mapError(displayViewRuntimeError(viewId, 'setShape'))
          )
        })

      yield* Effect.addFinalizer(() =>
        Effect.gen(function* () {
          const currentViews = yield* SynchronizedRef.get(views)
          yield* SynchronizedRef.set(views, new Map())
          yield* Effect.forEach(
            currentViews.values(),
            (entry) =>
              closeEntry(entry).pipe(
                Effect.zipRight(PubSub.shutdown(entry.pubsub))
              ),
            { discard: true }
          )
        })
      )

      return {
        setShape: (viewId, shape) =>
          Effect.gen(function* () {
            const currentViews = yield* SynchronizedRef.get(views)
            const existing = currentViews.get(viewId)
            if (existing && existing.failure === null && sameDisplayViewShape(existing.requestedShape, shape)) {
              return
            }

            const pubsub = existing?.pubsub ?? (yield* PubSub.unbounded<RuntimeDisplayViewUpdate>())
            const generation = (existing?.generation ?? 0) + 1
            const nextEntry = yield* makeEntry(viewId, shape, pubsub, generation)

            yield* SynchronizedRef.update(views, (latestViews) => {
              const nextViews = new Map(latestViews)
              nextViews.set(viewId, nextEntry)
              return nextViews
            })

            yield* PubSub.publish(pubsub, { _tag: 'snapshot', snapshot: nextEntry.snapshot })

            if (existing) {
              yield* closeEntry(existing)
            }
          }),

        stream: (viewId) =>
          Stream.unwrap(
            Effect.gen(function* () {
              const currentViews = yield* SynchronizedRef.get(views)
              const entry = currentViews.get(viewId)
              if (!entry) {
                return yield* new DisplayViewNotFoundError({ viewId })
              }

              const initial = Stream.succeed(entry.snapshot)
              const changes = entry.failure
                ? Stream.fail(entry.failure)
                : Stream.fromPubSub(entry.pubsub).pipe(
                    Stream.mapEffect(displayViewUpdateEffect)
                  )

              return Stream.concat(initial, changes)
            })
          ),

        snapshot: (viewId) =>
          Effect.gen(function* () {
            const currentViews = yield* SynchronizedRef.get(views)
            const entry = currentViews.get(viewId)
            if (!entry) {
              return yield* new DisplayViewNotFoundError({ viewId })
            }
            if (entry.failure) {
              return yield* entry.failure
            }
            return entry.snapshot
          }),

        close: (viewId) =>
          Effect.gen(function* () {
            const entry = yield* SynchronizedRef.modify(views, (currentViews) => {
              const current = currentViews.get(viewId)
              if (!current) return [null, currentViews] as const
              const nextViews = new Map(currentViews)
              nextViews.delete(viewId)
              return [current, nextViews] as const
            })
            if (!entry) return
            yield* closeEntry(entry)
            yield* PubSub.shutdown(entry.pubsub)
          }),
      } satisfies DisplayViewRuntimeService
    })
  )
