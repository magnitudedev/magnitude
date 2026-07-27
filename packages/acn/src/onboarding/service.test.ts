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
        reopenOnboardingFlow: () => Ref.set(stored, { completions: {} }),
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

  it("reopens a completed flow", async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const stored = yield* Ref.make<OnboardingConfig | null>({
        completions: {
          model_setup: { version: 1, completedAt: "2026-07-26T00:00:00.000Z" },
        },
      })
      const onboarding = makeOnboarding({
        getOnboardingConfig: () => Ref.get(stored),
        completeOnboardingFlow: (flowId, version, completedAt) => Ref.set(stored, {
          completions: { [flowId]: { version, completedAt } },
        }),
        reopenOnboardingFlow: () => Ref.set(stored, { completions: {} }),
      })
      yield* onboarding.reopen("model_setup")
      return yield* onboarding.state
    }))

    expect(result.flows.model_setup.required).toBe(true)
    expect(result.flows.model_setup.completedVersion).toBeNull()
  })
})
