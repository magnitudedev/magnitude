import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogReady,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import { selectedSlotModel } from "./model-slots"

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
})
