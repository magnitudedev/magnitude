import { Effect, Option } from "effect"
import { describe, expect, it } from "vitest"
import {
  LocalModelAvailableForDownload,
  LocalModelDownloading,
  LocalModelIdSchema,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/sdk"
import {
  isModelSlotLoadSatisfied,
  isModelSlotUnloadSatisfied,
  recoverRecentLocalSelection,
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
