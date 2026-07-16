import { describe, expect, it } from "vitest"
import { Schema } from "effect"
import {
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ProviderModelSchema,
  type ProviderId,
} from "./model"

describe("provider identity brands", () => {
  it("keeps provider, provider-model, and family IDs distinct", () => {
    const providerId = ProviderIdSchema.make("llamacpp")
    const providerModelId = ProviderModelIdSchema.make("lmp_model")
    const modelFamilyId = ModelFamilyIdSchema.make("qwen-3")

    // @ts-expect-error A provider-model ID cannot be used as a provider ID.
    const wrongProviderId: ProviderId = providerModelId

    const model = Schema.decodeUnknownSync(ProviderModelSchema)({
      providerId,
      providerModelId,
      modelFamilyId,
      displayName: "Qwen 3",
      contextWindow: 32_768,
      maxOutputTokens: 8_192,
      capabilities: { vision: false },
      availability: { _tag: "Available" },
      pricing: { input: 0, output: 0, cached_input: null },
      reasoningEfforts: ["none"],
    })

    expect(model.providerId).toBe("llamacpp")
    expect(wrongProviderId).toBe("lmp_model")
  })
})
