import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelSlotUnloadedLocalModel,
  PRIMARY_SLOT_ID,
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
  deriveOnboardingModelSetupView,
  onboardingModelSetupPlaceholder,
} from "./view-model"

const selection = {
  providerId: LOCAL_PROVIDER_ID,
  providerModelId: TEST_MODEL_ID,
  reasoningEffort: TEST_REASONING_EFFORT,
}

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
