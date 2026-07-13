/**
 * HydrationContext - Tracks whether we're in hydration mode
 */

import { Effect, Ref } from 'effect'

export class HydrationContext extends Effect.Service<HydrationContext>()('HydrationContext', {
  scoped: Effect.gen(function* () {
    const isHydratingRef = yield* Ref.make(false)

    return {
      isHydrating: (): Effect.Effect<boolean> => Ref.get(isHydratingRef),
      setHydrating: (value: boolean): Effect.Effect<void> => Ref.set(isHydratingRef, value),
    }
  }),
  dependencies: []
}) {}
