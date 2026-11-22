import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { localModelIsOwnedByMagnitude, localModelServingProfile } from "./local-model-projection"
import { LocalModelSchema, type LocalModel } from "./model-state"

const addedModel = (servingState: unknown) => Schema.decodeUnknownSync(LocalModelSchema)({
  _tag: "Custom",
  modelId: "hf:owner/repository/model.gguf",
  origin: { _tag: "Added", repository: "owner/repository", commit: "a".repeat(40) },
  drafterPending: false,
  drafterChoice: { _tag: "Auto" },
  presentation: { displayName: "Model", variantLabel: "Q4_K_M", sourceLabel: "Hugging Face", sourceUrls: [] },
  acquisitionState: {
    _tag: "Installing",
    progress: { stage: "downloading", completedBytes: 1, totalBytes: 4 },
  },
  servingState,
  storageBytes: 0,
})

describe("localModelServingProfile", () => {
  it("has no profile to show until the model has an assessment", () => {
    expect(localModelServingProfile(addedModel({ _tag: "Assessing", profile: { contextLength: 100_000 } }))).toEqual(Option.none())
    expect(localModelServingProfile(addedModel({
      _tag: "Failed",
      profile: { contextLength: 100_000 },
      failure: { code: "model_assessment_failed", message: "Failed", retryable: true },
    }))).toEqual(Option.none())
  })
})

describe("localModelIsOwnedByMagnitude", () => {
  const withInstallation = (tag: "Installed" | "External", ownership: "Magnitude" | "External" | "Mixed"): LocalModel => ({
    _tag: "Catalog",
    modelId: "lfm3-1.7b:gguf:q8",
    acquisitionState: {
      _tag: tag,
      installation: { _tag: "Resolved", installedBytes: 1, primaryPath: "/m.gguf", ownership },
      residencyState: { _tag: "Unloaded" },
    },
  } as unknown as LocalModel)

  it("distinguishes Magnitude-downloaded models from externally found or uninstalled models", () => {
    expect(localModelIsOwnedByMagnitude(withInstallation("Installed", "Magnitude"))).toBe(true)
    expect(localModelIsOwnedByMagnitude(withInstallation("Installed", "Mixed"))).toBe(true)
    expect(localModelIsOwnedByMagnitude(withInstallation("Installed", "External"))).toBe(false)
    expect(localModelIsOwnedByMagnitude(withInstallation("External", "External"))).toBe(false)
    expect(localModelIsOwnedByMagnitude(addedModel({ _tag: "Assessing", profile: { contextLength: 100_000 } }))).toBe(false)
  })
})
