import { Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
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
}
const readiness = {
  _tag: "Loadable" as const,
  allocation: {
    contextWindowTokens: 200_000,
    parallelSequences: 4,
    physicalContextTokens: 800_000,
    requiredSystemMemoryBytes: 60_000_000_000,
  },
}
const allocation = {
  contextWindowTokens: 200_000,
  parallelSequences: 3,
  physicalContextTokens: 600_000,
  memoryDomains: [],
}

describe("current local model derivation", () => {
  it("derives selection, readiness, and display identity from the slot alone", () => {
    const current = deriveCurrentLocalModel(Option.some(new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor,
      availability: { _tag: "Available" },
      readiness,
      instance: Option.none(),
      actions: ["Load"],
    })))

    expect(current).toMatchObject({
      _tag: "NotLoaded",
      displayName: "Canonical local model",
      preview: {
        _tag: "Some",
        value: { _tag: "Available", allocation: { parallelSequences: 4 } },
      },
    })
  })

  it("uses only the allocation owned by the exact ready instance", () => {
    const current = deriveCurrentLocalModel(Option.some(new ModelSlotConfiguredLocal({
      slotId: PRIMARY_SLOT_ID,
      selection,
      descriptor,
      availability: { _tag: "Available" },
      readiness,
      instance: Option.some({
        id: ModelInstanceIdSchema.make("instance"),
        configurationId: ModelServingConfigurationIdSchema.make("configuration"),
        lifecycle: { _tag: "Ready", allocation },
      }),
      actions: ["Stop"],
    })))

    expect(current).toMatchObject({
      _tag: "Running",
      allocation: { parallelSequences: 3, physicalContextTokens: 600_000 },
    })
  })
})
