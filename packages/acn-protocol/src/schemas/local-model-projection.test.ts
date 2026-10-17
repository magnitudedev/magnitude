import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { localModelServingProfile } from "./local-model-projection"
import { LocalModelSchema } from "./model-state"

const addedModel = (servingState: unknown) => Schema.decodeUnknownSync(LocalModelSchema)({
  _tag: "Custom",
  modelId: "hf:owner/repository/model.gguf",
  origin: { _tag: "Added", repository: "owner/repository", commit: "a".repeat(40) },
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
