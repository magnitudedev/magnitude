import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/acn-protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import {
  localModelSlotAvailability,
  modelSlotActions,
  projectModelInstance,
  selectableModelCapabilities,
} from "./model-slot-projection"
import type * as Generated from "@magnitudedev/icn-protocol/schemas"

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

  it("preserves structured low-memory failure facts", () => {
    const projected = projectModelInstance({
      id: "instance",
      configurationId: "configuration",
      lifecycle: {
        _tag: "Failed",
        failure: {
          _tag: "LowMemory",
          code: "low_memory",
          message: "not enough memory",
          retryable: true,
          requiredSystemMemoryBytes: 24,
          allocationHeadroomBytes: 20,
          systemReserveBytes: 2,
          loadBoundaryBytes: 26,
          minimumAdditionalAvailableBytes: 7,
          parallelSequences: 1,
        },
      },
    } as Generated.ModelInstance)

    expect(projected.lifecycle).toEqual({
      _tag: "Failed",
      failure: {
        _tag: "LowMemory",
        code: "low_memory",
        message: "not enough memory",
        retryable: true,
        requiredSystemMemoryBytes: 24,
        allocationHeadroomBytes: 20,
        systemReserveBytes: 2,
        loadBoundaryBytes: 26,
        minimumAdditionalAvailableBytes: 7,
        parallelSequences: 1,
      },
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
    expect(modelSlotActions({ _tag: "Pending" }, Option.none())).toEqual([])
  })

  it("keeps a durable local offering selected while its packages download", () => {
    expect(localModelSlotAvailability({
      catalogIdentityPending: false,
      offeringsReady: true,
      inventory: { _tag: "Ready" },
      offeringExists: true,
      installed: false,
    })).toEqual({
      _tag: "Unavailable",
      failure: {
        code: "local_model_not_installed",
        message: "The selected local model is not downloaded",
        retryable: true,
      },
    })
  })

  it("keeps incomplete local authority pending", () => {
    const input = {
      catalogIdentityPending: false,
      offeringsReady: true,
      inventory: { _tag: "Initializing" },
      offeringExists: true,
      installed: false,
    } as const
    expect(localModelSlotAvailability(input)).toEqual({ _tag: "Pending" })
    expect(localModelSlotAvailability({
      ...input,
      catalogIdentityPending: true,
      inventory: { _tag: "Ready" },
    })).toEqual({ _tag: "Pending" })
  })

  it("admits a durable offering before catalog publication", () => {
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
    const catalogCapabilities = {
      vision: false,
      tools: false,
      structuredOutput: false,
      reasoning: {
        supported: false,
        efforts: [],
        defaultEffort: Option.none(),
      },
    }
    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      undefined,
      { capabilities },
    )).toBe(capabilities)

    const catalogModel: ProviderModelCatalogEntry = {
      providerId: ProviderIdSchema.make("local"),
      providerModelId: ProviderModelIdSchema.make("test-configuration"),
      modelFamilyId: Option.none(),
      displayName: "Local model",
      variantLabel: Option.none(),
      supportedSlots: [SECONDARY_SLOT_ID],
      contextWindow: 4096,
      maxOutputTokens: 1024,
      capabilities: catalogCapabilities,
      availability: { _tag: "Available" },
      memory: Option.none(),
      pricing: Option.none(),
    }
    expect(selectableModelCapabilities(
      PRIMARY_SLOT_ID,
      catalogModel,
      { capabilities },
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
      { capabilities },
    )).toBe(capabilities)
  })

  it("does not use installed presentation to authorize a durable offering", () => {
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
      { capabilities },
    )).toBe(capabilities)
  })
})
