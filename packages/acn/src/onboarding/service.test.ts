import { describe, expect, it } from "vitest"
import { Effect, Ref } from "effect"
import type { OnboardingConfig } from "@magnitudedev/storage"
import { makeOnboarding } from "./service"

describe("Onboarding", () => {
  it("persists only the registered generic completion marker", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make<OnboardingConfig | null>(null)
      const onboarding = makeOnboarding({
        getOnboardingConfig: () => Ref.get(stored),
        completeOnboardingFlow: (flowId, version, completedAt) => Ref.set(stored, {
          completions: { [flowId]: { version, completedAt } },
        }),
      })
      const before = yield* onboarding.state
      yield* onboarding.complete("model_setup")
      const after = yield* onboarding.state
      return { before, after, stored: yield* Ref.get(stored) }
    }))

    expect(result.before.flows.model_setup).toMatchObject({
      currentVersion: 1,
      completedVersion: null,
      required: true,
    })
    expect(result.after.flows.model_setup).toMatchObject({
      currentVersion: 1,
      completedVersion: 1,
      required: false,
    })
    expect(result.after.flows.model_setup.completedAt).toEqual(expect.any(String))
    expect(result.stored).toEqual({
      completions: {
        model_setup: {
          version: 1,
          completedAt: result.after.flows.model_setup.completedAt,
        },
      },
    })
  })
})
