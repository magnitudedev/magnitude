import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelSlotConfiguredLocal,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import { deriveCurrentLocalModel } from "./current-local-model"

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("configuration"),
  reasoningEffort: ReasoningEffortSchema.make("none"),
}
const descriptor = {
  providerId: selection.providerId,
  providerModelId: selection.providerModelId,
  displayName: "Canonical local model",
  variantLabel: Option.none(),
}
const allocation = {
  contextWindowTokens: 200_000,
  parallelSequences: 3,
  physicalContextTokens: 600_000,
  memoryDomains: [],
}

describe("current local model derivation", () => {
  it("derives unloaded identity without fabricating allocation evidence", () => {
    const current = deriveCurrentLocalModel(Option.some(new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor,
      availability: { _tag: "Available" },
      residency: { _tag: "Unloaded" },
      actions: ["Load"],
    })))

    expect(current).toMatchObject({
      _tag: "NotLoaded",
      displayName: "Canonical local model",
      contextWindow: { _tag: "None" },
    })
  })

  it("uses only the allocation owned by the exact ready instance", () => {
    const current = deriveCurrentLocalModel(Option.some(new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor,
      availability: { _tag: "Available" },
      residency: {
        _tag: "Ready",
        instanceId: ModelInstanceIdSchema.make("instance"),
        allocation,
      },
      actions: ["Stop"],
    })))

    expect(current).toMatchObject({
      _tag: "Running",
      allocation: { parallelSequences: 3, physicalContextTokens: 600_000 },
    })
  })
})
