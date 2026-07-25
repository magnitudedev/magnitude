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
import {
  deriveLocalModelLoadActivity,
  isModelSlotConfigured,
  selectedSlotModel,
} from "./model-slots"

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
      memory: Option.none(),
      pricing: Option.none(),
    }
    const result = selectedSlotModel(
      new ProviderModelCatalogReady({ providers: [], models: [catalogModel] }),
      {
        slots: {
          primary: unloaded,
          secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
        },
        recentModelIds: { primary: [selection.providerModelId], secondary: [] },
        favoriteModels: [],
      },
      PRIMARY_SLOT_ID,
    )
    expect(Option.getOrThrow(result)).toMatchObject({ model: catalogModel, slot: unloaded })
  })

  it("keeps an assigned selection configured across runtime states", () => {
    expect(isModelSlotConfigured(
      new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
    )).toBe(false)
    expect(isModelSlotConfigured(unloaded)).toBe(true)
    expect(isModelSlotConfigured(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
      percentage: 25,
    }))).toBe(true)
    expect(isModelSlotConfigured(new ModelSlotReady({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))).toBe(true)
    expect(isModelSlotConfigured(new ModelSlotUnloadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
    }))).toBe(true)
    expect(isModelSlotConfigured(new ModelSlotBlocked({
      slotId: PRIMARY_SLOT_ID,
      selection,
      reason: { _tag: "ModelUnavailable", message: "Unavailable" },
    }))).toBe(true)
  })

  it("derives one shared model-loading presentation for a slot", () => {
    const slots = {
      slots: {
        primary: new ModelSlotLoadingLocalModel({
          slotId: PRIMARY_SLOT_ID,
          selection,
          percentage: 42,
        }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModelIds: { primary: [], secondary: [] },
      favoriteModels: [],
    }

    expect(deriveLocalModelLoadActivity(slots, PRIMARY_SLOT_ID)).toBe(slots.slots.primary)
    expect(deriveLocalModelLoadActivity(slots, SECONDARY_SLOT_ID)).toBeNull()
  })

  it("presents load and runtime pressure failures with the same low-memory message", () => {
    const slots = {
      slots: {
        primary: new ModelSlotBlocked({
          slotId: PRIMARY_SLOT_ID,
          selection,
          reason: {
            _tag: "LocalModelStoppedLowMemory" as const,
            error: {
              code: "low_memory",
              message: "internal detail",
              retryable: true,
            },
          },
        }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModelIds: { primary: [], secondary: [] },
      favoriteModels: [],
    }
    expect(deriveLocalModelLoadActivity(slots, PRIMARY_SLOT_ID)).toBe(slots.slots.primary)
  })
})
