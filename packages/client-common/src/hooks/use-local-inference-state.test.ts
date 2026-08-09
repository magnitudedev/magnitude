import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ProviderModelIdSchema,
} from "@magnitudedev/sdk"
import { findLocalConfigurationOffering } from "./use-local-inference-state"

describe("findLocalConfigurationOffering", () => {
  it("resolves only the exact target and configuration", () => {
    const targetId = ModelOfferingTargetIdSchema.make("target")
    const configurationId = ModelServingConfigurationIdSchema.make("configuration")
    const providerModelId = ProviderModelIdSchema.make("provider-model")
    const models = { models: [{ targetId, offerings: [{ configurationId, providerModelId }] }] }

    expect(Option.getOrNull(findLocalConfigurationOffering(models, targetId, configurationId)))
      .toBe(providerModelId)
    expect(Option.isNone(findLocalConfigurationOffering(
      models,
      targetId,
      ModelServingConfigurationIdSchema.make("different"),
    ))).toBe(true)
    expect(Option.isNone(findLocalConfigurationOffering(
      models,
      ModelOfferingTargetIdSchema.make("different"),
      configurationId,
    ))).toBe(true)
  })
})
