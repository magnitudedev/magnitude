import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalModelAvailableForDownload,
  LocalModelDownloading,
  LocalModelIdSchema,
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
  applyLocalModelLoadingStage,
  applyReplacedLocalModelStage,
  isModelSlotLoadSatisfied,
  isModelSlotUnloadSatisfied,
  recoverRecentLocalSelection,
  reconcileAvailableLocalSlot,
} from "./model-slot-coordinator"
import { transitionLocalInventoryEntry } from "./local-model-inventory"

const effort = ReasoningEffortSchema.make("high")
const localSelection = {
  providerId: ProviderIdSchema.make("local"),
  providerModelId: ProviderModelIdSchema.make("local:model"),
  reasoningEffort: effort,
}
const model = {
  localModelId: LocalModelIdSchema.make("model"),
  providerModelId: ProviderModelIdSchema.make("local:model"),
  modelFamilyId: Option.none(),
  displayName: "Model",
  family: "family",
  architecture: "Dense" as const,
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: true as const, efforts: [effort], defaultEffort: Option.some(effort) },
  },
  contextWindow: 4096,
  maxOutputTokens: 1024,
  quantization: "Q4",
  downloadBytes: 100,
  fit: { _tag: "Fits" as const, requiredBytes: 50, availableBytes: 100, memoryDomainIds: [] },
  recommendation: Option.none(),
}

describe("ACN model-state transitions", () => {
  it("publishes acquisition progress and completion through real FSM edges", async () => {
    const available = new LocalModelAvailableForDownload({ model })
    const downloading = await Effect.runPromise(transitionLocalInventoryEntry(Option.some(available), {
      kind: "Downloading",
      model,
      percentage: 35,
      completedBytes: 35,
      totalBytes: 100,
    }))
    const downloaded = await Effect.runPromise(transitionLocalInventoryEntry(Option.some(downloading), {
      kind: "Downloaded",
      model,
      downloadedBytes: 100,
    }))
    expect(downloading).toMatchObject({ _tag: "Downloading", percentage: 35 })
    expect(downloaded).toMatchObject({ _tag: "Downloaded", downloadedBytes: 100 })
  })

  it("rejects a skipped acquisition transition instead of fabricating intermediate states", async () => {
    const available = new LocalModelAvailableForDownload({ model })
    const result = await Effect.runPromise(Effect.either(transitionLocalInventoryEntry(Option.some(available), {
      kind: "Downloaded",
      model,
      downloadedBytes: 100,
    })))
    expect(result._tag).toBe("Left")
  })

  it("holds monotonic download progress", async () => {
    const downloading = new LocalModelDownloading({
      model,
      percentage: 60,
      completedBytes: 60,
      totalBytes: 100,
    })
    const stale = await Effect.runPromise(transitionLocalInventoryEntry(Option.some(downloading), {
      kind: "Downloading",
      model,
      percentage: 40,
      completedBytes: 40,
      totalBytes: 100,
    }))
    expect(stale).toMatchObject({ _tag: "Downloading", percentage: 60 })
  })

  it("treats repeated load and unload commands as already satisfied", () => {
    expect(isModelSlotLoadSatisfied(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
      percentage: 40,
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

  it("projects real ICN load fractions into monotonic slot percentages", () => {
    const unloaded = new ModelSlotUnloadedLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
    })
    const started = applyLocalModelLoadProgress(unloaded, 0)
    const advanced = started._tag === "Unassigned"
      ? started
      : applyLocalModelLoadProgress(started, 0.816)
    const stale = advanced._tag === "Unassigned"
      ? advanced
      : applyLocalModelLoadProgress(advanced, 0.2825)

    expect(started).toMatchObject({ _tag: "LoadingLocalModel", percentage: 0 })
    expect(advanced).toMatchObject({ _tag: "LoadingLocalModel", percentage: 82 })
    expect(stale).toMatchObject({ _tag: "LoadingLocalModel", percentage: 82 })
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
      percentage: 40,
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
    }
    const loading = {
      slots: {
        primary: applyLocalModelLoadingStage(unloading.slots.primary, replacementModelId, null),
        secondary: applyLocalModelLoadingStage(unloading.slots.secondary, replacementModelId, null),
      },
    }

    expect(Schema.is(ModelSlotsStateSchema)(unloading)).toBe(true)
    expect(Schema.is(ModelSlotsStateSchema)(loading)).toBe(true)
    expect(loading.slots.primary).toMatchObject({ _tag: "UnloadedLocalModel" })
    expect(loading.slots.secondary).toMatchObject({ _tag: "LoadingLocalModel", percentage: 0 })
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
        capabilities: model.capabilities,
        availability: { _tag: "Available" },
        pricing: Option.none(),
      }],
    )

    expect(Option.getOrThrow(recovered).providerModelId).toBe(recentModelId)
  })
})
