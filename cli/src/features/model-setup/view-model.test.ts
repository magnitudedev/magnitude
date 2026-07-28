import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelInstanceIdSchema,
  ModelSlotConfiguredLocal,
  PRIMARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  LOCAL_PROVIDER_ID,
  TEST_CONFIGURATION_ID,
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
    })).toBe(true)
  })

  it("stays inactive when onboarding is complete and setup was not forced", () => {
    expect(deriveModelSetupActive({
      forceSetup: false,
      onboardingRequired: false,
      completionSucceeded: false,
    })).toBe(false)
  })

  it("keeps forced setup active until the completed update succeeds", () => {
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: false,
    })).toBe(true)
  })

  it("completes forced setup after an explicit skip", () => {
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: true,
    })).toBe(false)
  })
})

describe("deriveOnboardingModelSetupView", () => {
  it("shows the chooser inside setup once recommendations are ready", () => {
    expect(deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      submittedProviderModelId: null,
      state: makeView({ ready: false }),
    })._tag).toBe("Choosing")
  })

  it("keeps an installed selection in setup until its exact instance is ready", () => {
    const base = makeView({ ready: false })
    const state = {
      ...base,
      slots: {
        ...base.slots,
        slots: {
          ...base.slots.slots,
          primary: new ModelSlotConfiguredLocal({
            slotId: PRIMARY_SLOT_ID,
            selection,
            descriptor: {
              providerId: LOCAL_PROVIDER_ID,
              providerModelId: TEST_MODEL_ID,
              displayName: "Qwen Test",
            },
            availability: { _tag: "Available" },
            instance: Option.none(),
            actions: ["Load"],
          }),
        },
      },
    }
    expect(deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      submittedProviderModelId: null,
      state,
    })._tag).toBe("Choosing")
    const view = deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      submittedProviderModelId: TEST_MODEL_ID,
      state,
    })
    expect(view).toMatchObject({
      _tag: "Activating",
      phase: "Preparing",
      displayName: "Qwen Test",
    })
    expect(onboardingModelSetupPlaceholder(view)).toBe("Preparing Qwen Test…")
  })

  it("shows Loading only when the selected exact instance is Loading", () => {
    const base = makeView({ ready: false })
    const primary = new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        displayName: "Qwen Test",
      },
      availability: { _tag: "Available" },
      instance: Option.some({
        id: ModelInstanceIdSchema.make("loading-instance"),
        configurationId: TEST_CONFIGURATION_ID,
        lifecycle: {
          _tag: "Loading",
          stage: "loading",
          progress: Option.some(0.25),
          plannedAllocation: Option.none(),
        },
      }),
      actions: ["Stop"],
    })
    const view = deriveOnboardingModelSetupView({
      active: true,
      onboardingRequired: true,
      submittedProviderModelId: TEST_MODEL_ID,
      state: {
        ...base,
        slots: {
          ...base.slots,
          slots: { ...base.slots.slots, primary },
        },
      },
    })
    expect(view).toMatchObject({ _tag: "Activating", phase: "Loading" })
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
      submittedProviderModelId: TEST_MODEL_ID,
      state,
    })
    expect(view).toMatchObject({ _tag: "Downloading", candidate: { displayName: "Qwen Test" } })
  })
})
