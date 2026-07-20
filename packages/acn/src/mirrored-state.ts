import { Context, Effect, Layer, PubSub, Ref, Schema, Stream } from "effect"
import type { MirroredSnapshot, MirroredStateInvalidation } from "@magnitudedev/protocol"

export interface MirroredStateTransition<State, Result> {
  readonly state: State
  readonly result: Result
  readonly changed?: boolean
}

export interface MirroredState<State> {
  readonly get: Effect.Effect<MirroredSnapshot<State>>
  readonly modify: <Result>(
    f: (state: State) => MirroredStateTransition<State, Result>,
  ) => Effect.Effect<{ readonly snapshot: MirroredSnapshot<State>; readonly result: Result }>
  readonly update: (f: (state: State) => State) => Effect.Effect<MirroredSnapshot<State>>
  readonly setIfChanged: (
    state: State,
    equivalent: (left: State, right: State) => boolean,
  ) => Effect.Effect<MirroredSnapshot<State>>
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
    const state = yield* Ref.make<MirroredSnapshot<State>>({ revision: 0, state: initial })
    const lock = yield* Effect.makeSemaphore(1)

    const modify: MirroredState<State>["modify"] = (f) => lock.withPermits(1)(Effect.gen(function* () {
      const previous = yield* Ref.get(state)
      const transition = f(previous.state)
      if (transition.changed === false) return { snapshot: previous, result: transition.result }

      const next: MirroredSnapshot<State> = {
        revision: previous.revision + 1,
        state: transition.state,
      }
      yield* Ref.set(state, next)
      const invalidation: MirroredStateInvalidation = {
        _tag: "changed",
        id: definition.id,
        revision: next.revision,
      }
      yield* stateChanges.publish(invalidation)
      return { snapshot: next, result: transition.result }
    }))

    return {
      get: Ref.get(state),
      modify,
      update: (f) => modify((current) => ({ state: f(current), result: undefined })).pipe(
        Effect.map(({ snapshot }) => snapshot),
      ),
      setIfChanged: (nextState, equivalent) => modify((current) => equivalent(current, nextState)
        ? { state: current, result: undefined, changed: false }
        : { state: nextState, result: undefined }).pipe(
          Effect.map(({ snapshot }) => snapshot),
        ),
    }
  })
