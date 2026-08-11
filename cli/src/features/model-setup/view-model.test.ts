import { Option } from "effect"
import { describe, expect, it } from "vitest"
import type { OnboardingModelOperation } from "@magnitudedev/client-common"
import {
  DownloadAttemptIdSchema,
  ModelInstanceIdSchema,
} from "@magnitudedev/sdk"
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

const operation = (state: OnboardingModelOperation): OnboardingModelOperation => state

describe("onboarding model setup projection", () => {
  it("preserves server-required and explicitly forced setup", () => {
    expect(deriveModelSetupActive({ forceSetup: false, onboardingRequired: true, completionSucceeded: true })).toBe(true)
    expect(deriveModelSetupActive({ forceSetup: true, onboardingRequired: false, completionSucceeded: false })).toBe(true)
    expect(deriveModelSetupActive({ forceSetup: true, onboardingRequired: false, completionSucceeded: true })).toBe(false)
  })

  it("shows the chooser while the workflow is idle", () => {
    const view = makeView()
    expect(deriveOnboardingModelSetupView({
      operationState: operation({ _tag: "Idle" }),
      models: view.models,
      slots: view.slots,
    })).toEqual({ _tag: "Choosing" })
  })

  it("shows zero-progress downloading as soon as installation is requested", () => {
    const model = makeCatalogModel()
    const view = makeView({ models: [model], ready: false })
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "RequestingInstallation",
        submission: { _tag: "InstallThenLoad", choice },
        cancellationRequested: false,
      }),
      models: view.models,
      slots: view.slots,
    })).toEqual({ _tag: "Downloading", model, starting: true, cancelling: false })
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
      operationState: operation({
        _tag: "DownloadAdmitted",
        submission: { _tag: "InstallThenLoad", choice },
        configurationId: TEST_CONFIGURATION_ID,
        providerModelId: TEST_MODEL_ID,
        attemptIds: [DownloadAttemptIdSchema.make("attempt")],
      }),
      models: view.models,
      slots: view.slots,
    })
    expect(state).toMatchObject({ _tag: "Downloading", model })
    expect(onboardingModelSetupPlaceholder(state)).toContain("Downloading")
  })

  it("does not present stale acquisition state as the newly admitted download", () => {
    const model = makeAcquiringModel({
      _tag: "Failed",
      attemptIds: [DownloadAttemptIdSchema.make("previous-attempt")],
      completedBytes: 0,
      totalBytes: 2 * GIB,
      failure: { code: "previous", message: "Previous download failed", retryable: true },
    })
    const view = makeView({ models: [model], ready: false })
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "DownloadAdmitted",
        submission: { _tag: "InstallThenLoad", choice },
        configurationId: TEST_CONFIGURATION_ID,
        providerModelId: TEST_MODEL_ID,
        attemptIds: [DownloadAttemptIdSchema.make("new-attempt")],
      }),
      models: view.models,
      slots: view.slots,
    })).toEqual({ _tag: "Downloading", model, starting: true, cancelling: false })
  })

  it("returns configuring after the same model becomes installed", () => {
    const model = makeCatalogModel()
    const view = makeView({ models: [model], ready: false })
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "DownloadAdmitted",
        submission: { _tag: "InstallThenLoad", choice },
        configurationId: TEST_CONFIGURATION_ID,
        providerModelId: TEST_MODEL_ID,
        attemptIds: [DownloadAttemptIdSchema.make("attempt")],
      }),
      models: view.models,
      slots: view.slots,
    })._tag).toBe("Downloading")

    const installed = {
      ...model,
      acquisitionState: { _tag: "Installed" as const, installedBytes: model.downloadBytes, origins: ["Magnitude"] as const },
    }
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "Assigning",
        submission: { _tag: "InstallThenLoad", choice },
        providerModelId: TEST_MODEL_ID,
        cancellationRequested: false,
      }),
      models: { ...view.models, models: [installed] },
      slots: view.slots,
    })).toMatchObject({ _tag: "Configuring", model: installed })
  })

  it("falls back to loading while an admitted provider identity propagates", () => {
    const view = makeView({ ready: false })
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "LoadAdmitted",
        submission: { _tag: "Load", choice: { providerModelId: TEST_MODEL_ID, displayName: "Qwen Test", reasoningEffort: TEST_REASONING_EFFORT } },
        providerModelId: TEST_MODEL_ID,
        instanceId: ModelInstanceIdSchema.make("not-yet-projected"),
      }),
      models: view.models,
      slots: view.slots,
    })).toMatchObject({ _tag: "Activating", phase: "Loading" })
  })

  it("derives cancellation presentation directly from the operation state", () => {
    const model = makeCatalogModel()
    const view = makeView({ models: [model], ready: false })
    expect(deriveOnboardingModelSetupView({
      operationState: operation({
        _tag: "RequestingDownloadCancellation",
        submission: { _tag: "InstallThenLoad", choice },
        configurationId: TEST_CONFIGURATION_ID,
        providerModelId: TEST_MODEL_ID,
        attemptIds: [DownloadAttemptIdSchema.make("attempt")],
      }),
      models: view.models,
      slots: view.slots,
    })).toEqual({ _tag: "Downloading", model, starting: true, cancelling: true })
  })
})
