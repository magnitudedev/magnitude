import { Effect, Queue, Stream, SubscriptionRef, type Equivalence, type Scope } from "effect"

export interface MaterializedProjection<A> {
  readonly get: Effect.Effect<A>
  readonly changes: Stream.Stream<A>
}

export const coalesceInvalidations = <R>(
  invalidations: Stream.Stream<unknown, never, R>,
): Effect.Effect<Stream.Stream<void>, never, R | Scope.Scope> => Effect.gen(function* () {
  const pending = yield* Queue.sliding<void>(1)
  yield* invalidations.pipe(
    Stream.runForEach(() => Queue.offer(pending, undefined)),
    Effect.ensuring(Queue.shutdown(pending)),
    Effect.forkScoped,
  )
  return Stream.fromQueue(pending)
})

export const materializeProjection = <A, R, R2>(options: {
  readonly project: Effect.Effect<A, never, R>
  readonly invalidations: Stream.Stream<unknown, never, R2>
  readonly equivalent: Equivalence.Equivalence<A>
}): Effect.Effect<MaterializedProjection<A>, never, R | R2 | Scope.Scope> => Effect.gen(function* () {
  const invalidations = yield* coalesceInvalidations(options.invalidations)
  const current = yield* SubscriptionRef.make(yield* options.project)
  yield* invalidations.pipe(
    Stream.runForEach(() => options.project.pipe(
      Effect.flatMap((next) => SubscriptionRef.get(current).pipe(
        Effect.flatMap((previous) => options.equivalent(previous, next)
          ? Effect.void
          : SubscriptionRef.set(current, next)),
      )),
    )),
    Effect.forkScoped,
  )

  return {
    get: SubscriptionRef.get(current),
    changes: current.changes.pipe(Stream.drop(1)),
  }
})
