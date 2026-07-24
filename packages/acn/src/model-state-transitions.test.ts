import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  ModelSlotBlocked,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotsStateSchema,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
} from "@magnitudedev/sdk"
import {
  applyLocalModelLoadProgress,
  applyReplacedLocalModelStage,
  isModelSlotLoadSatisfied,
  isModelSlotUnloadSatisfied,
  recoverRecentLocalSelection,
  reconcileAvailableLocalSlot,
} from "./model-slot-coordinator"

const effort = ReasoningEffortSchema.make("high")
const localSelection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("local:model"),
  reasoningEffort: effort,
}
const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: { supported: true as const, efforts: [effort], defaultEffort: Option.some(effort) },
}

describe("ACN model-state transitions", () => {
  it("keeps estimated load progress monotonic and reserves 100 for Ready", () => {
    const loading = new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
      percentage: 72,
    })

    expect(applyLocalModelLoadProgress(loading, 0.5)).toMatchObject({ percentage: 72 })
    expect(applyLocalModelLoadProgress(loading, 1)).toMatchObject({ percentage: 99 })
  })

  it("treats repeated load and unload commands as already satisfied", () => {
    expect(isModelSlotLoadSatisfied(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
      percentage: 0,
    }))).toBe(true)
    expect(isModelSlotLoadSatisfied(new ModelSlotReady({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
    }))).toBe(true)
    expect(isModelSlotUnloadSatisfied(new ModelSlotUnloadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
    }))).toBe(true)
    expect(isModelSlotUnloadSatisfied(new ModelSlotUnloadedLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
    }))).toBe(true)
  })

  it("unloads a ready local slot when the singleton backend is replaced", () => {
    const replacementModelId = ProviderModelIdSchema.make("local:replacement")
    const ready = new ModelSlotReady({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
    })
    const unloading = applyReplacedLocalModelStage(ready, replacementModelId, "unloading")
    const unloaded = applyReplacedLocalModelStage(unloading, replacementModelId, "unloaded")
    const replacement = new ModelSlotLoadingLocalModel({
      slotId: SECONDARY_SLOT_ID,
      selection: { ...localSelection, providerModelId: replacementModelId },
      percentage: 0,
    })

    expect(unloading).toMatchObject({ _tag: "UnloadingLocalModel" })
    expect(unloaded).toMatchObject({ _tag: "UnloadedLocalModel" })
    expect(applyReplacedLocalModelStage(replacement, replacementModelId, "unloaded")).toBe(replacement)
  })

  it("keeps every replacement-stage slot snapshot schema-valid", () => {
    const replacementModelId = ProviderModelIdSchema.make("local:replacement")
    const replacementSelection = { ...localSelection, providerModelId: replacementModelId }
    const ready = new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: localSelection })
    const replacement = new ModelSlotUnloadedLocalModel({
      slotId: SECONDARY_SLOT_ID,
      selection: replacementSelection,
    })
    const unloading = {
      slots: {
        primary: applyReplacedLocalModelStage(ready, replacementModelId, "unloading"),
        secondary: replacement,
      },
      recentModelIds: { primary: [], secondary: [] },
      favoriteModels: [],
    }
    expect(Schema.is(ModelSlotsStateSchema)(unloading)).toBe(true)
  })

  it("clears a transient availability block after the selected local model becomes valid", () => {
    const blocked = new ModelSlotBlocked({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
      reason: { _tag: "ProviderUnavailable", message: "Local model inventory is loading" },
    })

    expect(reconcileAvailableLocalSlot(PRIMARY_SLOT_ID, localSelection, Option.some(blocked))).toEqual(
      new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
    )
  })

  it("recovers a missing local selection from per-slot recency even when the model is unloaded", () => {
    const recentModelId = ProviderModelIdSchema.make("local:recent")
    const recovered = recoverRecentLocalSelection(
      PRIMARY_SLOT_ID,
      Option.some(localSelection),
      [recentModelId],
      [{
        providerId: ProviderIdSchema.make("local"),
        providerModelId: recentModelId,
        modelFamilyId: Option.none(),
        displayName: "Recent local model",
        supportedSlots: [PRIMARY_SLOT_ID],
        contextWindow: 200_000,
        maxOutputTokens: 32_768,
        runtimeMemoryBytes: Option.none(),
        capabilities,
        availability: { _tag: "Available" },
        pricing: Option.none(),
      }],
    )

    expect(Option.getOrThrow(recovered).providerModelId).toBe(recentModelId)
  })

  it("replaces an unsupported saved reasoning effort with the model default", () => {
    const recovered = recoverRecentLocalSelection(
      PRIMARY_SLOT_ID,
      Option.some({
        ...localSelection,
        reasoningEffort: ReasoningEffortSchema.make("medium"),
      }),
      [],
      [{
        providerId: ProviderIdSchema.make("local"),
        providerModelId: localSelection.providerModelId,
        modelFamilyId: Option.none(),
        displayName: "Local model",
        supportedSlots: [PRIMARY_SLOT_ID],
        contextWindow: 200_000,
        maxOutputTokens: 32_768,
        runtimeMemoryBytes: Option.none(),
        capabilities,
        availability: { _tag: "Available" },
        pricing: Option.none(),
      }],
    )

    expect(Option.getOrThrow(recovered).reasoningEffort).toBe(effort)
  })
})
