import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  modelSlotActions,
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
  projectModelResidency,
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
    const projected = projectModelResidency(instance)
    expect(projected._tag).toBe("Loading")
    if (projected._tag !== "Loading") return
    expect(projected.instanceId).toBe(ModelInstanceIdSchema.make("instance"))
    expect(projected.configurationId).toBe(
      ModelServingConfigurationIdSchema.make("configuration"),
    )
    expect(projected).toMatchObject({
      progress: Option.some(0.5),
    })
  })

  it("preserves structured low-memory failure facts", () => {
    const projected = projectModelResidency({
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

    expect(projected).toEqual({
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

  it("projects terminal instance history into current residency", () => {
    const stopped = (reason: "user_stop" | "memory_pressure") => projectModelResidency({
      id: "instance",
      configurationId: "configuration",
      lifecycle: { _tag: "Stopped", reason },
    } as Generated.ModelInstance)

    expect(stopped("user_stop")).toEqual({ _tag: "Unloaded" })
    expect(stopped("memory_pressure")).toMatchObject({
      _tag: "Failed",
      failure: { code: "low_memory", retryable: true },
    })
  })

  it("derives every model action from canonical residency", () => {
    const available = { _tag: "Available" as const }
    expect(modelSlotActions(available, { _tag: "Unloaded" })).toEqual(["Load"])
    expect(modelSlotActions(available, {
      _tag: "Loading",
      instanceId: ModelInstanceIdSchema.make("instance"),
      configurationId: ModelServingConfigurationIdSchema.make("configuration"),
      stage: "loading",
      progress: Option.none(),
      plannedAllocation: Option.none(),
    })).toEqual(["Stop"])
    expect(modelSlotActions(available, {
      _tag: "Failed",
      failure: { code: "failed", message: "failed", retryable: true },
    })).toEqual(["RetryLoad"])
    expect(modelSlotActions({
      _tag: "Unavailable",
      failure: { code: "offline", message: "offline", retryable: true },
    }, { _tag: "Unloaded" })).toEqual([])
    expect(modelSlotActions({ _tag: "Pending" }, { _tag: "Unloaded" })).toEqual([])
    expect(modelSlotActions(available, { _tag: "Requested" })).toEqual(["Stop"])
    expect(modelSlotActions(available, {
      _tag: "Failed",
      failure: { code: "failed", message: "failed", retryable: true },
    })).toEqual(["RetryLoad"])
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
