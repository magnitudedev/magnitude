import { Effect, Equivalence, Ref, Stream, SubscriptionRef } from "effect"

export interface IcnObservedSnapshot<A> {
  readonly revision: number
  readonly state: A
}

export interface IcnObservedState<A, E> {
  readonly get: Effect.Effect<IcnObservedSnapshot<A>>
  readonly changes: Stream.Stream<IcnObservedSnapshot<A>>
  readonly refresh: Effect.Effect<void, E>
}

export const makeIcnObservedState = <A, E>(
  initial: A,
  read: Effect.Effect<A, E>,
  equivalent: Equivalence.Equivalence<A>,
): Effect.Effect<IcnObservedState<A, E>> =>
  Effect.gen(function* () {
    const current = yield* SubscriptionRef.make<IcnObservedSnapshot<A>>({
      revision: 0,
      state: initial,
    })
    const refreshLock = yield* Effect.makeSemaphore(1)

    const refresh = refreshLock.withPermits(1)(Effect.gen(function* () {
      const nextState = yield* read
      const snapshot = yield* Ref.get(current)
      if (equivalent(snapshot.state, nextState)) return
      yield* SubscriptionRef.set(current, {
        revision: snapshot.revision + 1,
        state: nextState,
      })
    }))

    return {
      get: Ref.get(current),
      changes: current.changes,
      refresh,
    }
  })
