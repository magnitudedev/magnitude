import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import {
  LocalModelAvailableForDownload,
  LocalModelDownloadFailed,
  LocalModelDownloaded,
  LocalModelDownloading,
  LocalModelIdSchema,
  LocalModelInventoryEntryLifecycle,
  LocalModelInventoryEntryDetailsSchema,
  LocalModelInventoryFailed,
  LocalModelInventoryLifecycle,
  LocalModelInventoryLoading,
  LocalModelInventoryStateSchema,
  LocalModelInventoryReady,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceHardwareSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelCapabilitiesSchema,
  ModelSlotBlocked,
  ModelSlotLoadingLocalModel,
  ModelSlotLifecycle,
  ModelSlotReady,
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
  ModelSlotSchema,
  ModelSlotsStateSchema,
  PercentageSchema,
  PRIMARY_SLOT_ID,
  ProviderModelCatalogLifecycle,
  ProviderModelCatalogDegraded,
  ProviderModelCatalogLoading,
  ProviderModelCatalogReady,
  ProviderModelCatalogRefreshing,
  ProviderModelCatalogUnavailable,
  ProviderModelCatalogStateSchema,
  SECONDARY_SLOT_ID,
} from "./model-state"
import { ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from "@magnitudedev/ai"

const selection = {
  providerId: ProviderIdSchema.make("magnitude"),
  providerModelId: ProviderModelIdSchema.make("model"),
  reasoningEffort: ReasoningEffortSchema.make("high"),
}

const localSelection = {
  ...selection,
  providerId: ProviderIdSchema.make("local"),
}

const effort = ReasoningEffortSchema.make("high")
const model = {
  localModelId: LocalModelIdSchema.make("candidate"),
  providerModelId: ProviderModelIdSchema.make("local:candidate"),
  modelFamilyId: Option.none(),
  displayName: "Candidate",
  family: "Candidate",
  architecture: "Dense" as const,
  capabilities: Schema.decodeSync(ModelCapabilitiesSchema)({
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: true, efforts: [effort], defaultEffort: effort },
  }),
  contextWindow: 4096,
  maxOutputTokens: 1024,
  quantization: "Q4",
  downloadBytes: 100,
  fit: { _tag: "Fits" as const, requiredBytes: 10, availableBytes: 20, memoryDomainIds: [] },
  recommendation: Option.none(),
}

describe("product model FSMs", () => {
  const failure = { code: "test", message: "test", retryable: true }

  const assertMatrix = (
    lifecycle: { readonly transition: Function },
    sources: Readonly<Record<string, unknown>>,
    targetProperties: Readonly<Record<string, unknown>>,
    allowed: Readonly<Record<string, readonly string[]>>,
  ) => {
    for (const [sourceTag, source] of Object.entries(sources)) {
      for (const [targetTag, properties] of Object.entries(targetProperties)) {
        const transition = () => Reflect.apply(lifecycle.transition, undefined, [source, targetTag, properties])
        if (allowed[sourceTag]!.includes(targetTag)) expect(transition).not.toThrow()
        else expect(transition).toThrow("Invalid FSM transition")
      }
    }
  }

  it("accepts every allowed catalog transition and rejects every other edge", () => {
    const snapshot = { providers: [], models: [] }
    assertMatrix(ProviderModelCatalogLifecycle, {
      Loading: new ProviderModelCatalogLoading({}),
      Ready: new ProviderModelCatalogReady(snapshot),
      Refreshing: new ProviderModelCatalogRefreshing({ ...snapshot, failures: [] }),
      Degraded: new ProviderModelCatalogDegraded({ ...snapshot, failures: [] }),
      Unavailable: new ProviderModelCatalogUnavailable({ providers: [], failures: [] }),
    }, {
      Loading: {},
      Ready: snapshot,
      Refreshing: { ...snapshot, failures: [] },
      Degraded: { ...snapshot, failures: [] },
      Unavailable: { providers: [], failures: [] },
    }, {
      Loading: ["Ready", "Degraded", "Unavailable"],
      Ready: ["Refreshing"],
      Refreshing: ["Ready", "Degraded", "Unavailable"],
      Degraded: ["Refreshing"],
      Unavailable: ["Refreshing"],
    })
  })

  it("accepts every allowed inventory transition and rejects every other edge", () => {
    assertMatrix(LocalModelInventoryLifecycle, {
      Loading: new LocalModelInventoryLoading({}),
      Ready: new LocalModelInventoryReady({ entries: [] }),
      Failed: new LocalModelInventoryFailed({ error: failure }),
    }, {
      Loading: {},
      Ready: { entries: [] },
      Failed: { error: failure },
    }, {
      Loading: ["Ready", "Failed"],
      Ready: ["Failed"],
      Failed: ["Loading"],
    })
  })

  it("accepts every allowed inventory-entry transition and rejects every other edge", () => {
    assertMatrix(LocalModelInventoryEntryLifecycle, {
      AvailableForDownload: new LocalModelAvailableForDownload({ model }),
      Downloading: new LocalModelDownloading({ model, percentage: 10, completedBytes: 10, totalBytes: 100 }),
      Downloaded: new LocalModelDownloaded({ model, downloadedBytes: 100 }),
      DownloadFailed: new LocalModelDownloadFailed({ model, completedBytes: 10, totalBytes: 100, error: failure }),
    }, {
      AvailableForDownload: { model },
      Downloading: { model, percentage: 0, completedBytes: 0, totalBytes: 100 },
      Downloaded: { model, downloadedBytes: 100 },
      DownloadFailed: { model, completedBytes: 10, totalBytes: 100, error: failure },
    }, {
      AvailableForDownload: ["Downloading"],
      Downloading: ["Downloaded", "DownloadFailed"],
      Downloaded: ["AvailableForDownload"],
      DownloadFailed: ["Downloading"],
    })
  })

  it("accepts every allowed slot transition and rejects every other edge", () => {
    const reason = { _tag: "InvalidConfiguration" as const, message: "test" }
    assertMatrix(ModelSlotLifecycle, {
      Unassigned: new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
      UnloadedLocalModel: new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
      LoadingLocalModel: new ModelSlotLoadingLocalModel({ slotId: PRIMARY_SLOT_ID, selection: localSelection, percentage: 10 }),
      Ready: new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
      UnloadingLocalModel: new ModelSlotUnloadingLocalModel({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
      Blocked: new ModelSlotBlocked({ slotId: PRIMARY_SLOT_ID, selection: localSelection, reason }),
    }, {
      Unassigned: {},
      UnloadedLocalModel: { selection: localSelection },
      LoadingLocalModel: { selection: localSelection, percentage: 0 },
      Ready: { selection: localSelection },
      UnloadingLocalModel: { selection: localSelection },
      Blocked: { selection: localSelection, reason },
    }, {
      Unassigned: ["UnloadedLocalModel", "LoadingLocalModel", "Ready", "UnloadingLocalModel", "Blocked"],
      UnloadedLocalModel: ["Unassigned", "LoadingLocalModel", "Ready", "Blocked"],
      LoadingLocalModel: ["Unassigned", "Ready", "UnloadedLocalModel", "UnloadingLocalModel", "Blocked"],
      Ready: ["Unassigned", "UnloadingLocalModel", "UnloadedLocalModel", "Blocked"],
      UnloadingLocalModel: ["Unassigned", "UnloadedLocalModel", "LoadingLocalModel", "Ready", "Blocked"],
      Blocked: ["Unassigned", "UnloadedLocalModel", "LoadingLocalModel", "Ready"],
    })
  })

  it("enforces catalog refresh boundaries", () => {
    const ready = ProviderModelCatalogLifecycle.transition(
      new ProviderModelCatalogLoading({}),
      "Ready",
      { models: [], providers: [] },
    )
    expect(() => Reflect.apply(ProviderModelCatalogLifecycle.transition, undefined, [
      ready,
      "Degraded",
      { models: [], providers: [], failures: [] },
    ])).toThrow("Invalid FSM transition")
    const refreshing = ProviderModelCatalogLifecycle.transition(ready, "Refreshing", { failures: [] })
    expect(refreshing._tag).toBe("Refreshing")
  })

  it("retries an unavailable catalog through Refreshing", () => {
    const unavailable = new ProviderModelCatalogUnavailable({ providers: [], failures: [] })
    expect(ProviderModelCatalogLifecycle.transition(unavailable, "Refreshing", { models: [] })._tag)
      .toBe("Refreshing")
  })

  it("enforces slot transitions and supports same-state hold", () => {
    const unassigned = new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID })
    const ready = ModelSlotLifecycle.transition(unassigned, "Ready", { selection })
    expect(ready).toBeInstanceOf(ModelSlotReady)
    expect(ModelSlotLifecycle.hold(ready, { selection })._tag).toBe("Ready")
  })

  it("enforces inventory-entry transitions", () => {
    const available = new LocalModelAvailableForDownload({ model })
    const downloading = LocalModelInventoryEntryLifecycle.transition(available, "Downloading", {
      percentage: 25,
      completedBytes: 25,
      totalBytes: 100,
    })
    expect(downloading).toBeInstanceOf(LocalModelDownloading)
    const failed = LocalModelInventoryEntryLifecycle.transition(downloading, "DownloadFailed", {
      error: { code: "network", message: "failed", retryable: true },
    })
    expect(LocalModelInventoryEntryLifecycle.transition(failed, "Downloading", {
      percentage: 25,
      completedBytes: 25,
      totalBytes: 100,
    })._tag).toBe("Downloading")
  })

  it("supports every local slot phase with mandatory progress", () => {
    const unloaded = ModelSlotLifecycle.transition(
      new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID }),
      "UnloadedLocalModel",
      { selection: localSelection },
    )
    const loading = ModelSlotLifecycle.transition(unloaded, "LoadingLocalModel", { percentage: 0 })
    expect(loading).toBeInstanceOf(ModelSlotLoadingLocalModel)
    const progressed = ModelSlotLifecycle.hold(loading, { percentage: 42 })
    const ready = ModelSlotLifecycle.transition(progressed, "Ready", {})
    const unloading = ModelSlotLifecycle.transition(ready, "UnloadingLocalModel", {})
    expect(ModelSlotLifecycle.transition(unloading, "UnloadedLocalModel", {})._tag).toBe("UnloadedLocalModel")
  })
})

describe("local model recommendation evidence", () => {
  it("round-trips intent, explanation, and exact generation evidence", () => {
    const recommended = {
      ...model,
      recommendation: Option.some({
        intent: "balanced" as const,
        explanation: "Balances capability with responsive generation.",
        fidelityLabel: "Very high fidelity",
        fidelityEvidence: "Test evidence",
        repository: "owner/repo",
        revision: "commit",
        files: [],
        sourcePageUrl: "https://example.com/model",
        estimatedRuntimeBytes: 10,
        fitMarginBytes: 10,
        estimatedGeneration: Option.some({
          contextTokens: 100_000,
          lowerTokensPerSecond: 20,
          expectedTokensPerSecond: 25,
          upperTokensPerSecond: 30,
          confidence: "high" as const,
          method: "test-estimator",
        }),
      }),
    }
    const encoded = Schema.encodeSync(LocalModelInventoryEntryDetailsSchema)(recommended)
    expect(encoded.recommendation).toMatchObject({
      intent: "balanced",
      estimatedGeneration: { expectedTokensPerSecond: 25 },
    })
    const decoded = Schema.decodeUnknownSync(LocalModelInventoryEntryDetailsSchema)(encoded)
    expect(Option.getOrThrow(decoded.recommendation).explanation).toContain("responsive")
  })
})

describe("product model schemas", () => {
  it("rejects percentages outside the inclusive integer range", () => {
    expect(Schema.is(PercentageSchema)(0)).toBe(true)
    expect(Schema.is(PercentageSchema)(100)).toBe(true)
    expect(Schema.is(PercentageSchema)(100.1)).toBe(false)
    expect(Schema.is(PercentageSchema)(-1)).toBe(false)
  })

  it("brands both fixed slot IDs", () => {
    expect([PRIMARY_SLOT_ID, SECONDARY_SLOT_ID]).toEqual(["primary", "secondary"])
  })

  it("rejects inconsistent reasoning capabilities and invalid download totals", () => {
    expect(Schema.is(ModelCapabilitiesSchema)({
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: false, efforts: [effort], defaultEffort: Option.some(effort) },
    })).toBe(false)
    expect(Schema.is(LocalModelInventoryStateSchema)({
      _tag: "Ready",
      entries: [{ _tag: "Downloading", model, percentage: 100, completedBytes: 101, totalBytes: 100 }],
    })).toBe(false)
  })

  it("rejects duplicate catalog and inventory identities", () => {
    const catalogModel = {
      providerId: selection.providerId,
      providerModelId: selection.providerModelId,
      modelFamilyId: Option.none(),
      displayName: "Model",
      supportedSlots: [PRIMARY_SLOT_ID],
      contextWindow: 4096,
      maxOutputTokens: 1024,
      capabilities: model.capabilities,
      availability: { _tag: "Available" as const },
      pricing: Option.none(),
    }
    const provider = {
      providerId: selection.providerId,
      displayName: "Magnitude",
      authentication: "Authenticated" as const,
      availability: { _tag: "Available" as const },
    }
    expect(Schema.is(ProviderModelCatalogStateSchema)({
      _tag: "Ready",
      providers: [provider, provider],
      models: [catalogModel, catalogModel],
    })).toBe(false)
    expect(Schema.is(ProviderModelCatalogStateSchema)({
      _tag: "Unavailable",
      providers: [provider, provider],
      failures: [],
    })).toBe(false)
    const downloaded = new LocalModelDownloaded({ model, downloadedBytes: 100 })
    expect(Schema.is(LocalModelInventoryStateSchema)(new LocalModelInventoryReady({
      entries: [downloaded, downloaded],
    }))).toBe(false)
  })

  it("rejects a catalog model whose provider is absent", () => {
    expect(Schema.is(ProviderModelCatalogStateSchema)({
      _tag: "Ready",
      providers: [],
      models: [{
        providerId: selection.providerId,
        providerModelId: selection.providerModelId,
        modelFamilyId: Option.none(),
        displayName: "Model",
        supportedSlots: [PRIMARY_SLOT_ID],
        contextWindow: 4096,
        maxOutputTokens: 1024,
        capabilities: model.capabilities,
        availability: { _tag: "Available" },
        pricing: Option.none(),
      }],
    })).toBe(false)
  })

  it("rejects unresolved hardware references and duplicate hardware IDs", () => {
    const domainId = LocalInferenceMemoryDomainIdSchema.make("memory")
    const hardware = {
      platform: "Linux" as const,
      architecture: "X64" as const,
      processor: Option.none(),
      logicalCores: 8,
      totalSystemMemoryBytes: 100,
      availableSystemMemoryBytes: Option.some(50),
      memoryDomains: [{ memoryDomainId: domainId, kind: "System" as const, totalBytes: 100, stableCapacityBytes: 80, availableBytes: Option.some(50), sharesSystemMemory: true }],
      accelerators: [{ acceleratorId: LocalInferenceAcceleratorIdSchema.make("gpu"), name: "GPU", backend: "test", memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("missing") }],
    }
    expect(Schema.is(LocalInferenceHardwareSchema)(hardware)).toBe(false)
    expect(Schema.is(LocalInferenceHardwareSchema)({
      ...hardware,
      accelerators: [],
      memoryDomains: [...hardware.memoryDomains, ...hardware.memoryDomains],
    })).toBe(false)
  })

  it("rejects cloud selections in local-only slot states", () => {
    expect(Schema.is(ModelSlotSchema)(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection,
      percentage: 10,
    }))).toBe(false)
    expect(Schema.is(ModelSlotSchema)(new ModelSlotLoadingLocalModel({
      slotId: PRIMARY_SLOT_ID,
      selection: localSelection,
      percentage: 10,
    }))).toBe(true)
  })

  it("rejects two distinct active local models across slots", () => {
    const otherLocalSelection = {
      ...localSelection,
      providerModelId: ProviderModelIdSchema.make("local:other"),
    }
    expect(Schema.is(ModelSlotsStateSchema)({
      slots: {
        primary: new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
        secondary: new ModelSlotReady({ slotId: SECONDARY_SLOT_ID, selection: otherLocalSelection }),
      },
    })).toBe(false)
    expect(Schema.is(ModelSlotsStateSchema)({
      slots: {
        primary: new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
        secondary: new ModelSlotLoadingLocalModel({
          slotId: SECONDARY_SLOT_ID,
          selection: otherLocalSelection,
          percentage: 0,
        }),
      },
    })).toBe(false)
    expect(Schema.is(ModelSlotsStateSchema)({
      slots: {
        primary: new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection: localSelection }),
        secondary: new ModelSlotReady({ slotId: SECONDARY_SLOT_ID, selection: localSelection }),
      },
    })).toBe(true)
  })
})
