import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  PRIMARY_SLOT_ID,
  type LocalModelsState,
  type ProviderModelCatalogEntry,
  type ProviderModelCatalogState,
} from "@magnitudedev/acn-protocol"
import { ProviderIdSchema, ProviderModelIdSchema } from "@magnitudedev/sdk"
import { PROVIDER_ID as LOCAL_PROVIDER_ID } from "@magnitudedev/icn/provider"
import { projectModelCatalog } from "./model-catalog"

const remoteProviderId = ProviderIdSchema.make("remote-test")

const offering = (
  providerId: typeof remoteProviderId,
  providerModelId: string,
): ProviderModelCatalogEntry => ({
  providerId,
  providerModelId: ProviderModelIdSchema.make(providerModelId),
  modelFamilyId: Option.none(),
  displayName: providerModelId,
  variantLabel: Option.none(),
  supportedSlots: [PRIMARY_SLOT_ID],
  contextWindow: 8_192,
  maxOutputTokens: 1_024,
  memory: Option.none(),
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
  },
  availability: { _tag: "Available" },
  pricing: Option.none(),
})

const local = (overrides: Partial<LocalModelsState> = {}): LocalModelsState => ({
  inventoryState: { _tag: "Ready" },
  discoveryState: { _tag: "Ready", progress: [] },
  models: [],
  ...overrides,
})

describe("ACN model catalog projection", () => {
  it("keeps local model state visible while remote provider discovery initializes", () => {
    expect(projectModelCatalog({ _tag: "Loading" }, local())).toMatchObject({
      _tag: "Refreshing",
      providers: [],
      models: [],
      localInventoryState: { _tag: "Ready" },
    })
  })

  it("publishes remote rows once and does not duplicate native local offerings", () => {
    const providers: ProviderModelCatalogState = {
      _tag: "Ready",
      providers: [
        {
          providerId: remoteProviderId,
          displayName: "Remote",
          kind: "Hosted",
          authentication: "Authenticated",
          availability: { _tag: "Available" },
        },
        {
          providerId: LOCAL_PROVIDER_ID,
          displayName: "Local",
          kind: "Local",
          authentication: "NotRequired",
          availability: { _tag: "Available" },
        },
      ],
      models: [
        offering(remoteProviderId, "remote-model"),
        offering(LOCAL_PROVIDER_ID, "local-model"),
      ],
    }

    const result = projectModelCatalog(providers, local())

    expect(result._tag).toBe("Ready")
    if (result._tag !== "Ready") return
    expect(result.models).toEqual([{
      _tag: "Remote",
      offering: providers.models[0],
    }])
  })

  it("retains valid rows and exposes local projection degradation", () => {
    const providers: ProviderModelCatalogState = {
      _tag: "Ready",
      providers: [{
        providerId: remoteProviderId,
        displayName: "Remote",
        kind: "Hosted",
        authentication: "Authenticated",
        availability: { _tag: "Available" },
      }],
      models: [offering(remoteProviderId, "remote-model")],
    }
    const failure = { code: "inventory_unavailable", message: "inventory unavailable", retryable: true }

    const result = projectModelCatalog(providers, local({
      inventoryState: { _tag: "Degraded", failure },
    }))

    expect(result._tag).toBe("Degraded")
    if (result._tag !== "Degraded") return
    expect(result.models).toHaveLength(1)
    expect(result.failures).toContainEqual({ _tag: "CatalogFailure", message: failure.message })
  })
})
