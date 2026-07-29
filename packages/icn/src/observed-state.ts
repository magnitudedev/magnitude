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

export interface IcnMutableObservedState<A, E> extends IcnObservedState<A, E> {
  readonly update: (f: (state: A) => A) => Effect.Effect<void>
}

export const makeIcnObservedState = <A, E>(
  initial: A,
  read: Effect.Effect<A, E>,
  equivalent: Equivalence.Equivalence<A>,
): Effect.Effect<IcnMutableObservedState<A, E>> =>
  Effect.gen(function* () {
    const current = yield* SubscriptionRef.make({
      initialized: false,
      snapshot: {
        revision: 0,
        state: initial,
      },
    })
    const refreshLock = yield* Effect.makeSemaphore(1)

    const publish = (nextState: A) => Effect.gen(function* () {
      const previous = yield* SubscriptionRef.get(current)
      if (previous.initialized && equivalent(previous.snapshot.state, nextState)) return
      yield* SubscriptionRef.set(current, {
        initialized: true,
        snapshot: {
          revision: previous.snapshot.revision + 1,
          state: nextState,
        },
      })
    })

    const refresh = refreshLock.withPermits(1)(read.pipe(Effect.flatMap(publish)))
    const update = (f: (state: A) => A) => refreshLock.withPermits(1)(
      SubscriptionRef.get(current).pipe(
        Effect.map(({ snapshot }) => f(snapshot.state)),
        Effect.flatMap(publish),
      ),
    )

    return {
      get: SubscriptionRef.get(current).pipe(Effect.map(({ snapshot }) => snapshot)),
      changes: current.changes.pipe(Stream.map(({ snapshot }) => snapshot)),
      initialized: SubscriptionRef.get(current).pipe(Effect.map(({ initialized }) => initialized)),
      refresh,
      update,
    }
  })
