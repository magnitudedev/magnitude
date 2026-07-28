import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import {
  localModelAvailability,
  modelSlotActions,
  projectModelInstance,
  selectableModelCapabilities,
} from "./model-slot-projection"
import type { Generated } from "@magnitudedev/icn"

describe("model slot projection", () => {
  it("preserves exact instance and configuration identity from ICN", () => {
    const instance = {
      id: "instance",
      configurationId: "configuration",
      lifecycle: {
        _tag: "Loading",
        stage: "loading",
        progress: Option.some(0.5),
        plannedAllocation: Option.none(),
      },
    } as unknown as Generated.ModelInstance
    const projected = projectModelInstance(instance)
    expect(projected.id).toBe(ModelInstanceIdSchema.make("instance"))
    expect(projected.configurationId).toBe(
      ModelServingConfigurationIdSchema.make("configuration"),
    )
    expect(projected.lifecycle).toMatchObject({
      _tag: "Loading",
      progress: Option.some(0.5),
    })
  })

  it("derives every physical action from canonical instance lifecycle", () => {
    const available = { _tag: "Available" as const }
    expect(modelSlotActions(available, Option.none())).toEqual(["Load"])
    expect(modelSlotActions(available, Option.some({
      id: ModelInstanceIdSchema.make("instance"),
      configurationId: ModelServingConfigurationIdSchema.make("configuration"),
      lifecycle: {
        _tag: "Loading",
        stage: "loading",
        progress: Option.none(),
        plannedAllocation: Option.none(),
      },
    }))).toEqual(["Stop"])
    expect(modelSlotActions(available, Option.some({
      id: ModelInstanceIdSchema.make("instance"),
      configurationId: ModelServingConfigurationIdSchema.make("configuration"),
      lifecycle: {
        _tag: "Failed",
        failure: { code: "failed", message: "failed", retryable: true },
      },
    }))).toEqual(["RetryLoad"])
    expect(modelSlotActions({
      _tag: "Unavailable",
      failure: { code: "offline", message: "offline", retryable: true },
    }, Option.none())).toEqual([])
  })

  it("keeps a durable local offering selected while its packages download", () => {
    const catalogUnavailable = {
      _tag: "Unavailable" as const,
      failure: {
        code: "model_unavailable",
        message: "The catalog has not published the model yet",
        retryable: true,
      },
    }
    expect(localModelAvailability(catalogUnavailable, true, false)).toEqual({
      _tag: "Unavailable",
      failure: {
        code: "local_model_not_installed",
        message: "The selected local model is not downloaded",
        retryable: true,
      },
    })
  })

  it("admits an installed durable offering before catalog publication", () => {
    const packageId = ModelPackageIdSchema.make("package")
    const effort = ReasoningEffortSchema.make("none")
    const capabilities = {
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: {
        supported: true,
        efforts: [effort],
        defaultEffort: Option.some(effort),
      },
    }
    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      undefined,
      { capabilities, packageIds: [packageId] },
      new Set([packageId]),
    )).toBe(capabilities)

    const catalogModel: ProviderModelCatalogEntry = {
      providerId: ProviderIdSchema.make("local"),
      providerModelId: ProviderModelIdSchema.make("test-configuration"),
      modelFamilyId: Option.none(),
      displayName: "Local model",
      supportedSlots: [SECONDARY_SLOT_ID],
      contextWindow: 4096,
      maxOutputTokens: 1024,
      capabilities,
      availability: { _tag: "Available" },
      memory: Option.none(),
      pricing: Option.none(),
    }
    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      catalogModel,
      { capabilities, packageIds: [packageId] },
      new Set([packageId]),
    )).toBeUndefined()

    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      {
        ...catalogModel,
        supportedSlots: [PRIMARY_SLOT_ID],
        availability: {
          _tag: "Disabled",
          reason: "installation_unavailable",
        },
      },
      { capabilities, packageIds: [packageId] },
      new Set([packageId]),
    )).toBe(capabilities)
  })

  it("rejects an uninstalled durable offering", () => {
    const packageId = ModelPackageIdSchema.make("package")
    const effort = ReasoningEffortSchema.make("none")
    const capabilities = {
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: {
        supported: true,
        efforts: [effort],
        defaultEffort: Option.some(effort),
      },
    }
    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      undefined,
      { capabilities, packageIds: [packageId] },
      new Set(),
    )).toBeUndefined()
  })
})
