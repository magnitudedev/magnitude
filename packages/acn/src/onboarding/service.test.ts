import { describe, expect, it } from "vitest"
import { Effect, Layer, Ref } from "effect"
import {
  MagnitudeStorage,
  type MagnitudeStorageShape,
} from "@magnitudedev/storage"
import { AcnChangesLive } from "../changes"
import { makeOnboarding, Onboarding, OnboardingLive } from "./service"

describe("Onboarding", () => {
  it("defaults to incomplete and persists completion", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make({ completed: false })
      const onboarding = makeOnboarding({
        get: Ref.get(stored),
        update: (transition) => Ref.updateAndGet(stored, transition),
      })
      const before = yield* onboarding.state
      yield* onboarding.complete
      const after = yield* onboarding.state
      return { before, after, stored: yield* Ref.get(stored) }
    }))

    expect(result.before).toEqual({ completed: false })
    expect(result.after).toEqual({ completed: true })
    expect(result.stored).toEqual({ completed: true })
  })

  it("completion is idempotent and cannot return to incomplete", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make({ completed: true })
      const onboarding = makeOnboarding({
        get: Ref.get(stored),
        update: (transition) => Ref.updateAndGet(stored, transition),
      })
      yield* onboarding.complete
      return yield* onboarding.state
    }))

    expect(result).toEqual({ completed: true })
  })

  it("publishes updates through the onboarding mirror", async () => {
    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const stored = yield* Ref.make({ completed: false })
      const onboardingState = {
        get: Ref.get(stored),
        update: (transition: (current: { completed: boolean }) => { completed: boolean }) =>
          Ref.updateAndGet(stored, transition),
      }
      const storage = { onboarding: onboardingState } as unknown as MagnitudeStorageShape
      const layer = OnboardingLive.pipe(Layer.provide(Layer.mergeAll(
        Layer.succeed(MagnitudeStorage, storage),
        AcnChangesLive,
      )))
      return yield* Effect.gen(function* () {
        const onboarding = yield* Onboarding
        const before = yield* onboarding.snapshot
        yield* onboarding.complete
        const after = yield* onboarding.snapshot
        yield* onboarding.complete
        const afterRepeatedCompletion = yield* onboarding.snapshot
        return { before, after, afterRepeatedCompletion }
      }).pipe(Effect.provide(layer))
    })))

    expect(result.before).toMatchObject({
      revision: 0,
      state: { completed: false },
    })
    expect(result.after).toMatchObject({
      revision: 1,
      state: { completed: true },
    })
    expect(result.afterRepeatedCompletion).toEqual(result.after)
  })
})
