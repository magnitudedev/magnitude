import { Option } from "effect"
import { describe, expect, it } from "vitest"
import { DownloadAttemptIdSchema } from "@magnitudedev/sdk"
import {
  deriveModelSetupActive,
  deriveOnboardingModelSetupView,
  onboardingModelSetupPlaceholder,
} from "./view-model"
import {
  GIB,
  makeAcquiringModel,
  makeCatalogModel,
  makeView,
  TEST_CONFIGURATION_ID,
  TEST_MODEL_ID,
  TEST_REASONING_EFFORT,
} from "../local-inference/test-fixtures"

const choice = {
  configurationId: TEST_CONFIGURATION_ID,
  displayName: "Qwen Test",
  reasoningEffort: TEST_REASONING_EFFORT,
}

describe("onboarding model setup projection", () => {
  it("preserves server-required and explicitly forced setup", () => {
    expect(deriveModelSetupActive({ forceSetup: false, onboardingRequired: true, completionSucceeded: true })).toBe(true)
    expect(deriveModelSetupActive({ forceSetup: true, onboardingRequired: false, completionSucceeded: false })).toBe(true)
    expect(deriveModelSetupActive({ forceSetup: true, onboardingRequired: false, completionSucceeded: true })).toBe(false)
  })

  it("shows the chooser while the workflow is idle", () => {
    const view = makeView()
    expect(deriveOnboardingModelSetupView({
      active: true,
      submission: null,
      providerModelId: Option.none(),
      submitting: false,
      models: view.models,
      slots: view.slots,
    })).toEqual({ _tag: "Choosing" })
  })

  it("reads download progress from the submitted canonical model", () => {
    const model = makeAcquiringModel({
      _tag: "Downloading",
      attemptIds: [DownloadAttemptIdSchema.make("attempt")],
      stage: "downloading",
      completedBytes: GIB,
      totalBytes: 2 * GIB,
      bytesPerSecond: Option.none(),
    })
    const view = makeView({ models: [model], ready: false })
    const state = deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "ConfigureThenLoad", choice },
      providerModelId: Option.none(),
      submitting: true,
      models: view.models,
      slots: view.slots,
    })
    expect(state).toMatchObject({ _tag: "Downloading", model })
    expect(onboardingModelSetupPlaceholder(state)).toContain("Downloading")
  })

  it("returns configuring after the same model becomes installed", () => {
    const model = makeCatalogModel()
    const view = makeView({ models: [model], ready: false })
    expect(deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "ConfigureThenLoad", choice },
      providerModelId: Option.none(),
      submitting: true,
      models: view.models,
      slots: view.slots,
    })._tag).toBe("Downloading")

    const installed = {
      ...model,
      acquisitionState: { _tag: "Installed" as const, installedBytes: model.downloadBytes, origins: ["Magnitude"] as const },
    }
    expect(deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "ConfigureThenLoad", choice },
      providerModelId: Option.none(),
      submitting: true,
      models: { ...view.models, models: [installed] },
      slots: view.slots,
    })).toMatchObject({ _tag: "Configuring", model: installed })
  })

  it("falls back to loading while an admitted provider identity propagates", () => {
    const view = makeView({ ready: false })
    expect(deriveOnboardingModelSetupView({
      active: true,
      submission: { _tag: "Load", choice: { providerModelId: TEST_MODEL_ID, displayName: "Qwen Test", reasoningEffort: TEST_REASONING_EFFORT } },
      providerModelId: Option.some(TEST_MODEL_ID),
      submitting: true,
      models: view.models,
      slots: view.slots,
    })).toMatchObject({ _tag: "Activating", phase: "Loading" })
  })
})
