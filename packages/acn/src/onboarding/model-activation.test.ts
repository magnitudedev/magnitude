import { describe, expect, it } from "vitest"
import {
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotUnloadedLocalModel,
  PRIMARY_SLOT_ID,
  type OnboardingState,
} from "@magnitudedev/protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import { onboardingModelActivationAction } from "./model-activation"

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("local:test"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}

const onboarding = (required: boolean): OnboardingState => ({
  flows: {
    model_setup: {
      currentVersion: 1,
      completedVersion: required ? null : 1,
      completedAt: required ? null : "2026-07-27T00:00:00.000Z",
      required,
    },
  },
})

describe("onboardingModelActivationAction", () => {
  it("loads an installed selected model while onboarding remains open", () => {
    expect(onboardingModelActivationAction(
      onboarding(true),
      new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection }),
    )).toBe("load")
  })

  it("completes onboarding only after the selected model is ready", () => {
    expect(onboardingModelActivationAction(
      onboarding(true),
      new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection }),
    )).toBe("complete")
    expect(onboardingModelActivationAction(
      onboarding(true),
      new ModelSlotLoadingLocalModel({ slotId: PRIMARY_SLOT_ID, selection, percentage: 42 }),
    )).toBe("none")
  })

  it("does not mutate residency after onboarding is complete", () => {
    expect(onboardingModelActivationAction(
      onboarding(false),
      new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection }),
    )).toBe("none")
  })
})
