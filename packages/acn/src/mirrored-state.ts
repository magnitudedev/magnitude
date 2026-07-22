import { Context, Effect, Layer, PubSub, Schema, Stream, SubscriptionRef } from "effect"
import type { MirroredSnapshot, MirroredStateInvalidation } from "@magnitudedev/protocol"

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

export interface MirroredStateChangesApi {
  readonly publish: (event: MirroredStateInvalidation) => Effect.Effect<void>
  readonly stream: Stream.Stream<MirroredStateInvalidation>
}

export class MirroredStateChanges extends Context.Tag("MirroredStateChanges")<
  MirroredStateChanges,
  MirroredStateChangesApi
>() {}

export const MirroredStateChangesLive = Layer.effect(
  MirroredStateChanges,
  Effect.gen(function* () {
    const events = yield* PubSub.sliding<MirroredStateInvalidation>(256)
    return MirroredStateChanges.of({
      publish: (event) => PubSub.publish(events, event).pipe(Effect.asVoid),
      stream: Stream.fromPubSub(events),
    })
  }),
)

/** Authoritative versioned state with a coalescing invalidation-only stream. */
export const makeMirroredState = <const Id extends string, State, StateEncoded, StateRequirements>(
  definition: {
    readonly id: Id
    readonly stateSchema: Schema.Schema<State, StateEncoded, StateRequirements>
  },
  initial: NoInfer<State>,
): Effect.Effect<MirroredState<State>, never, MirroredStateChanges> =>
  Effect.gen(function* () {
    const stateChanges = yield* MirroredStateChanges
    const state = yield* SubscriptionRef.make<MirroredSnapshot<State>>({ revision: 0, state: initial })
    const lock = yield* Effect.makeSemaphore(1)

    const commit = (previous: MirroredSnapshot<State>, nextState: State) => Effect.uninterruptible(Effect.gen(function* () {
      const next: MirroredSnapshot<State> = {
        revision: previous.revision + 1,
        state: nextState,
      }
      yield* SubscriptionRef.set(state, next)
      yield* stateChanges.publish({
        _tag: "changed",
        id: definition.id,
        revision: next.revision,
      })
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

/**
 * Exposes an already authoritative, versioned source through the ACN mirror
 * invalidation channel without copying or re-versioning its state.
 */
export const bindMirroredState = <const Id extends string, State>(
  definition: { readonly id: Id },
  source: MirroredStateSource<State>,
) => Effect.gen(function* () {
  const stateChanges = yield* MirroredStateChanges
  const initial = yield* source.get
  yield* source.changes.pipe(
    Stream.dropWhile((snapshot) => snapshot.revision <= initial.revision),
    Stream.runForEach((snapshot) => stateChanges.publish({
      _tag: "changed",
      id: definition.id,
      revision: snapshot.revision,
    })),
    Effect.forkScoped,
  )
  return { get: source.get } satisfies MirroredStateReader<State>
})
