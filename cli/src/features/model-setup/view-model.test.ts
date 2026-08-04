import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  DownloadAttemptIdSchema,
  ModelInstanceIdSchema,
  ModelSlotConfiguredLocal,
  PRIMARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  LOCAL_PROVIDER_ID,
  TEST_CONFIGURATION_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
  TEST_TARGET_ID,
  makeCatalogCandidate,
  makeModel,
  makeView,
} from "../local-inference/test-fixtures"
import {
  deriveModelSetupActive,
  deriveOnboardingModelSetupView,
  onboardingModelSetupPlaceholder,
} from "./view-model"

const choice = {
  providerModelId: TEST_MODEL_ID,
  displayName: "Qwen Test",
  reasoningEffort: TEST_REASONING_EFFORT,
}

describe("deriveModelSetupActive", () => {
  it("preserves server-required and explicitly forced setup", () => {
    expect(deriveModelSetupActive({
      forceSetup: false,
      onboardingRequired: true,
      completionSucceeded: false,
    })).toBe(true)
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: false,
    })).toBe(true)
    expect(deriveModelSetupActive({
      forceSetup: true,
      onboardingRequired: false,
      completionSucceeded: true,
    })).toBe(false)
  })
})

describe("deriveOnboardingModelSetupView", () => {
  it("shows the chooser while the workflow is idle", () => {
    const state = makeView({ ready: false })
    const view = deriveOnboardingModelSetupView({
      active: true,
      submission: null,
      submitting: false,
      ...state,
    })
    expect(view._tag).toBe("Choosing")
    expect(onboardingModelSetupPlaceholder(view)).toBe("Select a model to start coding…")
  })

  it("shows loading immediately, before slot mirrors propagate", () => {
    const state = makeView({ ready: false })
    const view = deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "Load", choice },
      submitting: true,
      ...state,
    })
    expect(view).toMatchObject({
      _tag: "Activating",
      displayName: "Qwen Test",
      phase: "Loading",
    })
    expect(onboardingModelSetupPlaceholder(view)).toBe("Loading Qwen Test…")
  })

  it("uses the exact selected instance lifecycle once available", () => {
    const base = makeView({ ready: false })
    const primary = new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        reasoningEffort: TEST_REASONING_EFFORT,
      },
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
      submission: { _tag: "Load", choice },
      submitting: true,
      ...base,
      slots: { ...base.slots, slots: { ...base.slots.slots, primary } },
    })
    expect(view).toMatchObject({ _tag: "Activating", phase: "Loading" })
  })

  it("shows download immediately, before progress mirrors propagate", () => {
    const candidate = makeCatalogCandidate()
    const base = makeView({ models: [makeModel()], ready: false })
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
      submission: {
        _tag: "DownloadThenLoad",
        choice: { ...choice, targetId: TEST_TARGET_ID },
      },
      submitting: true,
      ...state,
    })
    expect(view).toMatchObject({ _tag: "Downloading", candidate: { displayName: "Qwen Test" } })
  })

  it("projects an authoritative download failure", () => {
    const candidate = makeCatalogCandidate({
      download: {
        _tag: "Failed",
        attemptIds: [DownloadAttemptIdSchema.make("download_failed")],
        completedBytes: 16,
        totalBytes: 16,
        failure: { code: "interrupted", message: "Download was interrupted", retryable: true },
      },
      preparation: { _tag: "NotDownloaded" },
    })
    const base = makeView({ models: [makeModel({ download: candidate.download })] })
    const view = deriveOnboardingModelSetupView({
      active: true,
      submission: {
        _tag: "DownloadThenLoad",
        choice: { ...choice, targetId: TEST_TARGET_ID },
      },
      submitting: false,
      ...base,
      models: {
        ...base.models,
        recommendations: {
          _tag: "Ready",
          entries: [],
          catalog: [candidate],
          progress: [],
        },
      },
    })
    expect(view._tag).toBe("DownloadFailed")
    expect(onboardingModelSetupPlaceholder(view)).toBe("Couldn’t download Qwen Test")
  })

  it("returns to choosing when an externally stopped load has settled", () => {
    const base = makeView({ ready: false })
    const primary = new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        reasoningEffort: TEST_REASONING_EFFORT,
      },
      descriptor: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        displayName: "Qwen Test",
      },
      availability: { _tag: "Available" },
      instance: Option.some({
        id: ModelInstanceIdSchema.make("stopped-instance"),
        configurationId: TEST_CONFIGURATION_ID,
        lifecycle: { _tag: "Stopped", reason: "user_stop" },
      }),
      actions: ["Load"],
    })
    for (const submitting of [true, false]) {
      const view = deriveOnboardingModelSetupView({
        active: true,
        submission: { _tag: "Load", choice },
        submitting,
        ...base,
        slots: { ...base.slots, slots: { ...base.slots.slots, primary } },
      })

      expect(view._tag).toBe("Choosing")
      expect(onboardingModelSetupPlaceholder(view)).toBe("Select a model to start coding…")
    }
  })

  it("renders the authoritative stopping lifecycle while cancellation settles", () => {
    const base = makeView({ ready: false })
    const primary = new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        reasoningEffort: TEST_REASONING_EFFORT,
      },
      descriptor: {
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        displayName: "Qwen Test",
      },
      availability: { _tag: "Available" },
      instance: Option.some({
        id: ModelInstanceIdSchema.make("stopping-instance"),
        configurationId: TEST_CONFIGURATION_ID,
        lifecycle: {
          _tag: "Stopping",
          reason: "user_stop",
          allocation: { _tag: "Planned", allocation: Option.none() },
        },
      }),
      actions: [],
    })
    const view = deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "Load", choice },
      submitting: true,
      ...base,
      slots: { ...base.slots, slots: { ...base.slots.slots, primary } },
    })

    expect(view).toMatchObject({ _tag: "Activating", phase: "Stopping" })
    expect(onboardingModelSetupPlaceholder(view)).toBe("Stopping Qwen Test…")
  })

})
