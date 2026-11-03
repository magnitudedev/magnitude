import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { localModelChatCapabilities, localModelListedAs, localModelServes, localModelServingProfile } from "./local-model-projection"
import { LocalModelSchema } from "./model-state"

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

describe("localModelServes", () => {
  const metadata = { format: "gguf", quantization: "Q8_0", quantizationName: "Q8_0", architecture: "qwen35", storageBytes: 1 }
  const chat = { serves: "chat", vision: false, tools: false, structuredOutput: false, reasoning: { supported: false, efforts: [] } }
  const supported = (capabilities: unknown) => ({ _tag: "Assessed", metadata, capabilities, assessment: fits })
  const refused = (serves?: string) => ({ _tag: "Assessed", metadata, assessment: unsupported, ...serves === undefined ? {} : { serves } })
  const fits = {
    _tag: "Fits",
    assessmentId: "assessment",
    environmentId: "environment",
    profile: { contextLength: 4096 },
    memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0 },
    performance: { _tag: "Estimated", samples: [{ contextTokens: 4096, estimatedTokensPerSecond: 10 }] },
  }
  const unsupported = {
    _tag: "Unsupported",
    environmentId: "environment",
    reasons: [{ _tag: "Unreadable" }],
    detail: "unreadable",
  }

  it("reads a custom model's kind from its assessment, and knows none before it or for a file that doesn't read", () => {
    const decider = addedModel(supported({ serves: "decisions", vision: false }))
    expect(localModelServes(decider)).toEqual(Option.some("decisions"))
    expect(localModelServes(addedModel(supported(chat)))).toEqual(Option.some("chat"))
    expect(localModelServes(addedModel({ _tag: "Assessing", profile: { contextLength: 4096 } }))).toEqual(Option.none())
    expect(localModelServes(addedModel(refused("decisions")))).toEqual(Option.some("decisions"))
    expect(localModelServes(addedModel(refused()))).toEqual(Option.none())
  })

  it("lists a model under its own kind, and under every kind while its kind isn't known", () => {
    const decider = addedModel(supported({ serves: "decisions", vision: false }))
    expect(localModelListedAs(decider, "decisions")).toBe(true)
    expect(localModelListedAs(decider, "chat")).toBe(false)
    const refusedDecider = addedModel(refused("decisions"))
    expect(localModelListedAs(refusedDecider, "chat")).toBe(false)
    const assessing = addedModel({ _tag: "Assessing", profile: { contextLength: 4096 } })
    expect(localModelListedAs(assessing, "chat")).toBe(true)
    expect(localModelListedAs(assessing, "decisions")).toBe(true)
  })

  it("gives chat capabilities only to a chat model, and none to one it can't run", () => {
    expect(localModelChatCapabilities(addedModel(supported(chat)))).toEqual(Option.some({
      vision: false, tools: false, structuredOutput: false, reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
    }))
    expect(localModelChatCapabilities(addedModel(supported({ serves: "decisions", vision: true })))).toEqual(Option.none())
    expect(localModelChatCapabilities(addedModel(refused("chat")))).toEqual(Option.none())
    expect(() => addedModel(supported({ serves: "decisions", vision: false, tools: true }))).not.toThrow()
    expect(addedModel(supported({ serves: "decisions", vision: false, tools: true })).servingState).not.toHaveProperty("capabilities.tools")
  })
})
