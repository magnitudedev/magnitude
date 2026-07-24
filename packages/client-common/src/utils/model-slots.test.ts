import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelSlotBlocked,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogReady,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import { isModelSlotUsableForMessages, selectedSlotModel } from "./model-slots"

const selection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("local:model"),
  reasoningEffort: ReasoningEffortSchema.make("high"),
}

const unloaded = new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection })

describe("model slot selection", () => {
  it("joins an unloaded local selection to its catalog model", () => {
    const catalogModel = {
      providerId: selection.providerId,
      providerModelId: selection.providerModelId,
      modelFamilyId: Option.none(),
      displayName: "Local model",
      supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
      contextWindow: 4096,
      maxOutputTokens: 1024,
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: {
          supported: true,
          efforts: [selection.reasoningEffort],
          defaultEffort: Option.some(selection.reasoningEffort),
        },
      },
      availability: { _tag: "Available" as const },
      pricing: Option.none(),
    }
    const result = selectedSlotModel(
      new ProviderModelCatalogReady({ providers: [], models: [catalogModel] }),
      {
        slots: {
          primary: unloaded,
          secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
        },
      },
      PRIMARY_SLOT_ID,
    )
    expect(Option.getOrThrow(result)).toMatchObject({ model: catalogModel, slot: unloaded })
  })

  it("only treats slots that can admit a message as usable", () => {
    expect(isModelSlotUsableForMessages(
      new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
    )).toBe(false)
    expect(isModelSlotUsableForMessages(unloaded)).toBe(true)
    expect(isModelSlotUsableForMessages(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
      percentage: 25,
    }))).toBe(true)
    expect(isModelSlotUsableForMessages(new ModelSlotReady({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))).toBe(true)
    expect(isModelSlotUsableForMessages(new ModelSlotUnloadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))).toBe(false)
    expect(isModelSlotUsableForMessages(new ModelSlotBlocked({
      slotId: PRIMARY_SLOT_ID,
      selection,
      reason: { _tag: "ModelUnavailable", message: "Unavailable" },
    }))).toBe(false)
  })
})
