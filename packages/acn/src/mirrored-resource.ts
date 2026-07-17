import { Effect, PubSub, Ref, Stream } from "effect"
import type { MirroredResourceInvalidation, MirroredSnapshot } from "@magnitudedev/protocol"

export interface MirroredResource<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
  readonly changes: Stream.Stream<MirroredResourceInvalidation>
  readonly update: (f: (state: State) => State) => Effect.Effect<MirroredSnapshot<State>>
  readonly setIfChanged: (
    state: State,
    equivalent: (left: State, right: State) => boolean,
  ) => Effect.Effect<MirroredSnapshot<State>>
}

/** Authoritative versioned state with an invalidation-only change stream. */
export const makeMirroredResource = <State>(initial: State): Effect.Effect<MirroredResource<State>> =>
  Effect.gen(function* () {
    const state = yield* Ref.make<MirroredSnapshot<State>>({ revision: 0, state: initial })
    const changes = yield* PubSub.unbounded<MirroredResourceInvalidation>()
    const lock = yield* Effect.makeSemaphore(1)

    const update = (f: (current: State) => State) => lock.withPermits(1)(Effect.gen(function* () {
      const previous = yield* Ref.get(state)
      const next: MirroredSnapshot<State> = {
        revision: previous.revision + 1,
        state: f(previous.state),
      }
      yield* Ref.set(state, next)
      const invalidation: MirroredResourceInvalidation = { _tag: "changed", revision: next.revision }
      yield* PubSub.publish(changes, invalidation)
      return next
    }))

    return {
      get: Ref.get(state),
      changes: Stream.fromPubSub(changes),
      update,
      setIfChanged: (nextState, equivalent) => lock.withPermits(1)(Effect.gen(function* () {
        const previous = yield* Ref.get(state)
        if (equivalent(previous.state, nextState)) return previous
        const next = { revision: previous.revision + 1, state: nextState }
        yield* Ref.set(state, next)
        const invalidation: MirroredResourceInvalidation = { _tag: "changed", revision: next.revision }
        yield* PubSub.publish(changes, invalidation)
        return next
      })),
    }
  })
