import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  modelSlotActions,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  type ProviderModelCatalogEntry,
} from "@magnitudedev/acn-protocol"
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/providers/client"
import {
  localModelSlotAvailability,
  selectableModelCapabilities,
} from "./model-slot-projection"

describe("model slot projection", () => {
  it("derives every model action from canonical residency", () => {
    const available = { _tag: "Available" as const }
    expect(modelSlotActions(available, { _tag: "Unloaded" })).toEqual(["Load"])
    expect(modelSlotActions(available, {
      _tag: "Loading",
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

  it("reports a missing local offering as unavailable", () => {
    expect(localModelSlotAvailability({
      catalogIdentityPending: false,
      offeringsReady: true,
      offeringExists: false,
    })).toEqual({
      _tag: "Unavailable",
      failure: {
        code: "local_offering_unavailable",
        message: "The selected local configuration is unavailable",
        retryable: true,
      },
    })
  })

  it("keeps incomplete catalog authority pending", () => {
    const input = {
      catalogIdentityPending: true,
      offeringsReady: true,
      offeringExists: true,
    } as const
    expect(localModelSlotAvailability(input)).toEqual({ _tag: "Pending" })
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
