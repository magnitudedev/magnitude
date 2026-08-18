import { Effect, Schema, SubscriptionRef } from "effect"
import {
  ModelStateSchema,
  type ModelState,
  type StateHandle,
} from "@magnitudedev/storage"

export const makeTestModelState = (
  initial: ModelState,
): Effect.Effect<StateHandle<ModelState, never>> => Effect.gen(function* () {
  const state = yield* SubscriptionRef.make(initial)
  const lock = yield* Effect.makeSemaphore(1)
  const equivalent = Schema.equivalence(ModelStateSchema)

  const modify: StateHandle<ModelState, never>["modify"] = (transition) =>
    lock.withPermits(1)(Effect.gen(function* () {
      const current = yield* SubscriptionRef.get(state)
      const [result, next] = transition(current)
      if (!equivalent(current, next)) yield* SubscriptionRef.set(state, next)
      return result
    }))

  return {
    get: SubscriptionRef.get(state),
    changes: state.changes,
    modify,
    update: (transition) => modify((current) => {
      const next = transition(current)
      return [next, next] as const
    }),
  }
})
