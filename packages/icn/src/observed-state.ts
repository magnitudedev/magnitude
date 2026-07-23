import { Effect, Equivalence, Stream, SubscriptionRef } from "effect"

export interface IcnObservedSnapshot<A> {
  readonly revision: number
  readonly state: A
}

export interface IcnObservedState<A, E> {
  readonly get: Effect.Effect<IcnObservedSnapshot<A>>
  readonly changes: Stream.Stream<IcnObservedSnapshot<A>>
  readonly initialized: Effect.Effect<boolean>
  readonly refresh: Effect.Effect<void, E>
}

export const makeIcnObservedState = <A, E>(
  initial: A,
  read: Effect.Effect<A, E>,
  equivalent: Equivalence.Equivalence<A>,
): Effect.Effect<IcnObservedState<A, E>> =>
  Effect.gen(function* () {
    const current = yield* SubscriptionRef.make({
      initialized: false,
      snapshot: {
        revision: 0,
        state: initial,
      },
    })
    const refreshLock = yield* Effect.makeSemaphore(1)

    const refresh = refreshLock.withPermits(1)(Effect.gen(function* () {
      const nextState = yield* read
      yield* SubscriptionRef.modify(current, (previous) =>
        previous.initialized && equivalent(previous.snapshot.state, nextState)
          ? [undefined, previous]
          : [undefined, {
              initialized: true,
              snapshot: {
                revision: previous.snapshot.revision + 1,
                state: nextState,
              },
            }])
    }))

    return {
      get: SubscriptionRef.get(current).pipe(Effect.map(({ snapshot }) => snapshot)),
      changes: current.changes.pipe(Stream.map(({ snapshot }) => snapshot)),
      initialized: SubscriptionRef.get(current).pipe(Effect.map(({ initialized }) => initialized)),
      refresh,
    }
  })
