import { Effect, Stream, SubscriptionRef, type Equivalence } from "effect"
import type { MirroredSnapshot } from "@magnitudedev/acn-protocol"
import { AcnChanges } from "./changes"

export interface MirroredStateTransition<State, Result> {
  readonly state: State
  readonly result: Result
  readonly changed?: boolean
}

export interface MirroredState<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
  readonly changes: Stream.Stream<MirroredSnapshot<State>>
  readonly modify: <Result>(
    f: (state: State) => MirroredStateTransition<State, Result>,
  ) => Effect.Effect<{ readonly snapshot: MirroredSnapshot<State>; readonly result: Result }>
  readonly update: (f: (state: State) => State) => Effect.Effect<MirroredSnapshot<State>>
  readonly setIfChanged: (
    state: State,
    equivalent: (left: State, right: State) => boolean,
  ) => Effect.Effect<MirroredSnapshot<State>>
}

export interface MirroredStateSource<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
  readonly changes: Stream.Stream<MirroredSnapshot<State>>
}

export interface MirroredStateReader<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
}

export interface ObservedState<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
  readonly changes: Stream.Stream<MirroredSnapshot<State>>
  readonly setIfChanged: (
    state: State,
    equivalent: Equivalence.Equivalence<State>,
  ) => Effect.Effect<void>
}

/** The query a versioned state backs: its change pokes name this query. */
export interface MirroredStateDefinition {
  readonly name: string
}

/**
 * Authoritative versioned state whose commits publish `{ query, revision }`
 * pokes on the ACN change registry. Clients reread the named query.
 */
export const makeMirroredState = <State>(
  definition: MirroredStateDefinition,
  initial: State,
): Effect.Effect<MirroredState<State>, never, AcnChanges> =>
  Effect.gen(function* () {
    const changes = yield* AcnChanges
    const state = yield* SubscriptionRef.make<MirroredSnapshot<State>>({ revision: 0, state: initial })
    const lock = yield* Effect.makeSemaphore(1)

    const commit = (previous: MirroredSnapshot<State>, nextState: State) => Effect.uninterruptible(Effect.gen(function* () {
      const next: MirroredSnapshot<State> = {
        revision: previous.revision + 1,
        state: nextState,
      }
      yield* SubscriptionRef.set(state, next)
      yield* changes.publish({ query: definition.name, revision: next.revision })
      return next
    }))

    const modify: MirroredState<State>["modify"] = (f) => lock.withPermits(1)(Effect.gen(function* () {
      const previous = yield* SubscriptionRef.get(state)
      const transition = f(previous.state)
      if (transition.changed === false) return { snapshot: previous, result: transition.result }
      const next = yield* commit(previous, transition.state)
      return { snapshot: next, result: transition.result }
    }))

    return {
      get: SubscriptionRef.get(state),
      changes: state.changes,
      modify,
      update: (f) => lock.withPermits(1)(Effect.gen(function* () {
        const previous = yield* SubscriptionRef.get(state)
        return yield* commit(previous, f(previous.state))
      })),
      setIfChanged: (nextState, equivalent) => lock.withPermits(1)(Effect.gen(function* () {
        const previous = yield* SubscriptionRef.get(state)
        return equivalent(previous.state, nextState)
          ? previous
          : yield* commit(previous, nextState)
      })),
    }
  })

export const makeObservedState = <State>(initial: State): Effect.Effect<ObservedState<State>> =>
  Effect.gen(function* () {
    const state = yield* SubscriptionRef.make<MirroredSnapshot<State>>({
      revision: 0,
      state: initial,
    })
    const lock = yield* Effect.makeSemaphore(1)
    return {
      get: SubscriptionRef.get(state),
      changes: state.changes,
      setIfChanged: (nextState, equivalent) => lock.withPermits(1)(Effect.gen(function* () {
        const current = yield* SubscriptionRef.get(state)
        if (equivalent(current.state, nextState)) return
        yield* SubscriptionRef.set(state, {
          revision: current.revision + 1,
          state: nextState,
        })
      })),
    }
  })

/**
 * Exposes an already authoritative, versioned source through the ACN change
 * registry without copying or re-versioning its state.
 */
export const bindMirroredState = <State>(
  definition: MirroredStateDefinition,
  source: MirroredStateSource<State>,
) => Effect.gen(function* () {
  const changes = yield* AcnChanges
  const initial = yield* source.get
  yield* source.changes.pipe(
    Stream.dropWhile((snapshot) => snapshot.revision <= initial.revision),
    Stream.runForEach((snapshot) => changes.publish({ query: definition.name, revision: snapshot.revision })),
    Effect.forkScoped,
  )
  return { get: source.get } satisfies MirroredStateReader<State>
})
