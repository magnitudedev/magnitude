import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelSlotUnloadedLocalModel,
  PRIMARY_SLOT_ID,
  ProviderModelIdSchema,
} from "@magnitudedev/sdk"
import {
  LOCAL_PROVIDER_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
  makeCatalogCandidate,
  makeModel,
  makeView,
} from "../local-inference/test-fixtures"
import {
  deriveModelSetupActive,
  deriveOnboardingModelSetupView,
  onboardingModelSetupPlaceholder,
} from "./view-model"

const selection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId: TEST_MODEL_ID,
  reasoningEffort: TEST_REASONING_EFFORT,
}

describe("deriveModelSetupActive", () => {
  it("preserves ordinary server-required onboarding", () => {
    expect(deriveModelSetupActive({
      forceSetup: false,
      onboardingRequired: true,
      completionSucceeded: false,
      selectedProviderModelId: null,
      primary: null,
    })).toBe(true)
  })

  it("stays inactive when onboarding is complete and setup was not forced", () => {
    expect(deriveModelSetupActive({
      forceSetup: false,
      onboardingRequired: false,
      completionSucceeded: false,
      selectedProviderModelId: null,
      primary: makeView().slots.slots.primary,
    })).toBe(false)
  })

  it("keeps forced setup active while selection success is ahead of mirrored state", () => {
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: false,
      selectedProviderModelId: ProviderModelIdSchema.make("local:new-selection"),
      primary: makeView().slots.slots.primary,
    })).toBe(true)
  })

  it("completes forced setup only when the exact selected model is ready", () => {
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: false,
      selectedProviderModelId: TEST_MODEL_ID,
      primary: makeView().slots.slots.primary,
    })).toBe(false)
  })

  it("completes forced setup after an explicit skip", () => {
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: true,
      selectedProviderModelId: null,
      primary: null,
    })).toBe(false)
  })
})

describe("deriveOnboardingModelSetupView", () => {
  it("shows the chooser inside setup once recommendations are ready", () => {
    expect(deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      state: makeView({ ready: false }),
    })._tag).toBe("Choosing")
  })

  it("keeps an installed selection in setup until residency is ready", () => {
    const base = makeView({ ready: false })
    const state = {
      ...base,
      slots: {
        ...base.slots,
        slots: {
          ...base.slots.slots,
          primary: new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection }),
        },
      },
    }
    const view = deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      state,
    })
    expect(view).toMatchObject({ _tag: "Loading", displayName: "Qwen Test" })
    expect(onboardingModelSetupPlaceholder(view)).toBe("Loading Qwen Test…")
  })

  it("projects the selected candidate download from authoritative model state", () => {
    const candidate = makeCatalogCandidate({
      download: {
        _tag: "Downloading",
        stage: "downloading",
        completedBytes: 1,
        totalBytes: 2,
        bytesPerSecond: Option.none(),
      },
      preparation: { _tag: "NotDownloaded" },
    })
    const base = makeView({ models: [makeModel({
      download: candidate.download,
      preparation: { _tag: "NotDownloaded" },
    })] })
    const state = {
      ...base,
      models: {
        ...base.models,
        recommendations: {
          ...base.models.recommendations,
          catalog: [candidate],
        },
      },
    }
    const view = deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      state,
    })
    expect(view).toMatchObject({ _tag: "Downloading", candidate: { displayName: "Qwen Test" } })
  })
})
