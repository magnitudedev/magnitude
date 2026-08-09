import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ProviderModelIdSchema,
  RecommendationIdSchema,
} from "@magnitudedev/sdk"
import {
  buildInstalledLocalModelChoices,
  buildLocalInferenceSelections,
  describeLocalHardware,
  describeLocalHardwareSummary,
  formatDownloadBytes,
  localInferenceProgressLines,
  performanceRange,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"
import {
  GIB,
  makeCatalogCandidate,
  makeHardware,
  makeModel,
  makeRecommendation,
  makeView,
} from "./test-fixtures"

const selectionsFor = (view: ReturnType<typeof makeView>) =>
  buildLocalInferenceSelections(view.models, view.catalog, view.slots)

describe("local inference onboarding presentation", () => {
  it("formats model artifacts in decimal gigabytes", () => {
    expect(formatDownloadBytes(73_395_172_000)).toBe("73.4 GB")
  })

  it("bounds the displayed speed range to a 50K catalog profile", () => {
    expect(performanceRange(makeCatalogCandidate({
      profile: { contextLength: 50_000 },
      performance: [
        {
          contextTokens: 25_000,
          lowerTokensPerSecond: 26,
          estimatedTokensPerSecond: 30,
          upperTokensPerSecond: 34,
          confidence: "moderate",
        },
        {
          contextTokens: 50_000,
          lowerTokensPerSecond: 16,
          estimatedTokensPerSecond: 20,
          upperTokensPerSecond: 24,
          confidence: "moderate",
        },
      ],
    }))).toEqual({
      lowerContext: 25_000,
      upperContext: 50_000,
      lowerTokensPerSecond: 20,
      upperTokensPerSecond: 30,
    })
  })
})

describe("local inference selection view model", () => {
  it("projects a newly downloaded target without creating a provider offering", () => {
    const before = makeView({
      ready: false,
      models: [makeModel({
        offerings: [],
        download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
      })],
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
        availability: { _tag: "NotDownloaded" },
      })],
    })
    const after = makeView({
      ready: false,
      models: [makeModel({ offerings: [] })],
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    })

    expect(buildInstalledLocalModelChoices(before.models, before.catalog, before.slots)).toEqual([])
    const choices = buildInstalledLocalModelChoices(after.models, after.catalog, after.slots)
    expect(choices).toHaveLength(1)
    expect(choices[0]).toMatchObject({
      id: `installed:${makeModel().targetId}`,
      configurationId: makeCatalogCandidate().configurationId,
    })
    expect(Option.isNone(choices[0]!.providerModel)).toBe(true)
  })

  it("keeps one target-centered choice when its provider offering is created", () => {
    const withoutOffering = makeView({
      ready: false,
      models: [makeModel({ offerings: [] })],
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    })
    const withOffering = makeView({
      ready: false,
      catalogCandidates: [makeCatalogCandidate({
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    })

    const before = buildInstalledLocalModelChoices(
      withoutOffering.models,
      withoutOffering.catalog,
      withoutOffering.slots,
    )
    const after = buildInstalledLocalModelChoices(
      withOffering.models,
      withOffering.catalog,
      withOffering.slots,
    )
    expect(before).toHaveLength(1)
    expect(after).toHaveLength(1)
    expect(after[0]!.id).toBe(before[0]!.id)
    expect(Option.map(after[0]!.providerModel, ({ providerModelId }) => providerModelId))
      .toEqual(Option.some(ProviderModelIdSchema.make("configuration_test")))
  })

  it("presents cumulative recommendation progress with authoritative counts and timing", () => {
    expect(localInferenceProgressLines([
      {
        id: "hardware",
        status: {
          _tag: "Completed",
          startedAtMs: 1_000,
          durationMs: 1_250,
          cached: false,
        },
        completedItems: Option.some(1),
        totalItems: Option.some(1),
        estimatedRemainingMs: Option.none(),
      },
      {
        id: "inventory",
        status: {
          _tag: "Completed",
          startedAtMs: 1_500,
          durationMs: 500,
          cached: false,
        },
        completedItems: Option.some(2),
        totalItems: Option.some(2),
        estimatedRemainingMs: Option.none(),
      },
      {
        id: "assessment",
        status: { _tag: "Running", startedAtMs: 2_000 },
        completedItems: Option.some(8),
        totalItems: Option.some(28),
        estimatedRemainingMs: Option.some(5_000),
      },
      {
        id: "assessment",
        status: {
          _tag: "Completed",
          startedAtMs: 2_000,
          durationMs: 7_600,
          cached: false,
        },
        completedItems: Option.some(20),
        totalItems: Option.some(20),
        estimatedRemainingMs: Option.none(),
      },
    ])).toEqual([
      {
        id: "hardware",
        state: "completed",
        label: "Detected hardware",
        metadata: "",
      },
      {
        id: "inventory",
        state: "completed",
        label: "Found 2 downloaded models",
        metadata: "",
      },
      {
        id: "assessment",
        state: "running",
        label: "Assessing models for this machine",
        metadata: " · 8/28 · about 5s left",
      },
      {
        id: "assessment",
        state: "completed",
        label: "Assessed 20 models for this machine",
        metadata: " · 8s",
      },
    ])
  })

  it("keeps cache reuse and recommendation timing out of presentation", () => {
    expect(localInferenceProgressLines([
      {
        id: "assessment",
        status: {
          _tag: "Completed",
          startedAtMs: 1_000,
          durationMs: 0,
          cached: true,
        },
        completedItems: Option.some(20),
        totalItems: Option.some(20),
        estimatedRemainingMs: Option.none(),
      },
      {
        id: "recommendations",
        status: {
          _tag: "Completed",
          startedAtMs: 1_000,
          durationMs: 500,
          cached: true,
        },
        completedItems: Option.some(4),
        totalItems: Option.some(4),
        estimatedRemainingMs: Option.none(),
      },
    ])).toEqual([
      {
        id: "assessment",
        state: "completed",
        label: "Assessed 20 models for this machine",
        metadata: "",
      },
      {
        id: "recommendations",
        state: "completed",
        label: "Prepared 4 recommendations",
        metadata: "",
      },
    ])
  })

  it("presents unified memory from the hardware contract", () => {
    const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("unified")
    const hardware = makeHardware({
      platform: "MacOS",
      architecture: "Arm64",
      productName: Option.some("MacBook Pro"),
      processor: Option.some("Apple M4 Max"),
      totalSystemMemoryBytes: 64 * GIB,
      accelerators: [{
        acceleratorId: LocalInferenceAcceleratorIdSchema.make("metal"),
        name: "MTL0",
        backend: "Metal",
        memoryDomainId,
      }],
      memoryDomains: [{
        memoryDomainId,
        kind: "UnifiedMemory",
        totalBytes: 64 * GIB,
        stableCapacityBytes: 52 * GIB,
        availableBytes: Option.none(),
        sharesSystemMemory: true,
      }],
    })

    expect(describeLocalHardware(hardware)).toEqual({
      system: {
        name: "Apple M4 Max",
        details: [
          "macOS · ARM64 · 16 logical CPU cores",
          "64.0 GiB unified memory · Metal GPU acceleration",
        ],
      },
      accelerators: [],
    })
  })

  it("uses the accelerator identity for a unified NVIDIA system", () => {
    const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("unified")
    const hardware = makeHardware({
      platform: "Linux",
      architecture: "Arm64",
      productName: Option.some("DGX Spark"),
      processor: Option.some("CPU"),
      logicalCores: 20,
      totalSystemMemoryBytes: 128 * GIB,
      accelerators: [{
        acceleratorId: LocalInferenceAcceleratorIdSchema.make("cuda"),
        name: "NVIDIA GB10",
        backend: "CUDA",
        memoryDomainId,
      }],
      memoryDomains: [{
        memoryDomainId,
        kind: "UnifiedMemory",
        totalBytes: 128 * GIB,
        stableCapacityBytes: 116 * GIB,
        availableBytes: Option.none(),
        sharesSystemMemory: true,
      }],
    })

    expect(describeLocalHardware(hardware).system).toEqual({
      name: "DGX Spark · NVIDIA GB10",
      details: [
        "Linux · ARM64 · 20 logical CPU cores",
        "128.0 GiB unified memory · CUDA GPU acceleration",
      ],
    })
    expect(describeLocalHardwareSummary(hardware)).toEqual([{
      name: "DGX Spark · NVIDIA GB10",
      details: ["Linux ARM64", "20 cores", "128 GiB unified", "CUDA"],
    }])
  })

  it("presents unified hardware as one compact physical-domain row", () => {
    const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("unified")
    expect(describeLocalHardwareSummary(makeHardware({
      platform: "MacOS",
      architecture: "Arm64",
      processor: Option.some("Apple M4 Max"),
      totalSystemMemoryBytes: 64 * GIB,
      accelerators: [{
        acceleratorId: LocalInferenceAcceleratorIdSchema.make("metal"),
        name: "MTL0",
        backend: "Metal",
        memoryDomainId,
      }],
      memoryDomains: [{
        memoryDomainId,
        kind: "UnifiedMemory",
        totalBytes: 64 * GIB,
        stableCapacityBytes: 52 * GIB,
        availableBytes: Option.none(),
        sharesSystemMemory: true,
      }],
    }))).toEqual([{
      name: "Apple M4 Max",
      details: ["macOS ARM64", "16 cores", "64 GiB unified", "Metal"],
    }])
  })

  it("separates system and discrete accelerator hardware", () => {
    expect(describeLocalHardwareSummary(makeHardware({
      platform: "Linux",
      architecture: "X64",
      productName: Option.none(),
      processor: Option.some("AMD Ryzen 9"),
      totalSystemMemoryBytes: 64 * GIB,
    }))).toEqual([
      {
        name: "AMD Ryzen 9",
        details: ["Linux x86-64", "16 cores", "64 GiB RAM"],
      },
      {
        name: "Test GPU",
        details: ["24 GiB VRAM", "CUDA"],
      },
    ])
  })

  it("identifies CPU-only inference without inventing an accelerator", () => {
    expect(describeLocalHardwareSummary(makeHardware({
      accelerators: [],
      memoryDomains: [],
    }))).toEqual([{
      name: "Test CPU",
      details: ["Linux x86-64", "16 cores", "64 GiB RAM", "CPU inference"],
    }])
  })

  it("classifies the downloaded model selected by a ready slot as running", () => {
    expect(selectionsFor(makeView())[0]?.kind).toBe("running")
  })

  it("shows an assessed installed target before it has a provider offering", () => {
    const configurationId = ModelServingConfigurationIdSchema.make("configuration_default")
    const selections = selectionsFor(makeView({
      ready: false,
      models: [makeModel({ offerings: [] })],
      catalogCandidates: [makeCatalogCandidate({
        configurationId,
        profile: { contextLength: 131_072 },
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    }))

    expect(selections).toHaveLength(1)
    expect(selections[0]).toMatchObject({ kind: "stored" })
    expect(selections[0]?.kind === "stored" && selections[0].configurationId)
      .toBe(configurationId)
    expect(Option.isNone(selections[0]!.providerModelId)).toBe(true)
  })

  it("keeps the preferred configuration when another offering exists", () => {
    const preferredConfigurationId = ModelServingConfigurationIdSchema.make("configuration_default")
    const selections = selectionsFor(makeView({
      ready: false,
      catalogCandidates: [makeCatalogCandidate({
        configurationId: preferredConfigurationId,
        profile: { contextLength: 100_000 },
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    }))

    expect(selections).toHaveLength(1)
    expect(selections[0]?.kind === "stored" && selections[0].configurationId)
      .toBe(preferredConfigurationId)
    expect(Option.isNone(selections[0]!.providerModelId)).toBe(true)
  })

  it("keeps the selected offering when the preferred catalog configuration differs", () => {
    const view = makeView({
      catalogCandidates: [makeCatalogCandidate({
        configurationId: ModelServingConfigurationIdSchema.make("configuration_preferred"),
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    })

    const [choice] = buildInstalledLocalModelChoices(view.models, view.catalog, view.slots)
    expect(choice?.configurationId).toBe(ModelServingConfigurationIdSchema.make("configuration_test"))
    expect(choice?.running).toBe(true)
    expect(Option.map(choice!.providerModel, ({ providerModelId }) => providerModelId))
      .toEqual(Option.some(ProviderModelIdSchema.make("configuration_test")))
  })

  it("shows the configured context for a downloaded catalog model", () => {
    const selection = selectionsFor(makeView({
      models: [makeModel({ maximumContextLength: 262_144 })],
      providerContextWindow: 100_000,
      catalogCandidates: [makeCatalogCandidate({
        profile: { contextLength: 100_000 },
        download: { _tag: "Downloaded", installedBytes: 16 * GIB },
        availability: { _tag: "Available" },
      })],
    }))[0]

    expect(selection).toBeDefined()
    expect(selectionMetadata(selection!)).toContain("100K ctx")
    expect(selectionMetadata(selection!)).not.toContain("256K ctx")
  })

  it("keeps recommendations actionable without duplicating target state", () => {
    const model = makeModel({
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
    })
    const selections = selectionsFor(makeView({
      models: [model],
      recommendations: [makeRecommendation()],
      ready: false,
    }))
    expect(selections).toHaveLength(1)
    expect(selections[0]?.kind).toBe("recommendation")
    expect(selectionMetadata(selections[0]!)).toContain("Q4_K_M")
  })

  it("orders recommendation intents for comparison rather than by model name", () => {
    const recommendation = (
      intent: "balanced" | "best_quality" | "fastest" | "lightweight",
      index: number,
    ) => {
      const targetId = ModelOfferingTargetIdSchema.make(`target_${index}`)
      return makeRecommendation({
        id: RecommendationIdSchema.make(`recommendation_${intent}`),
        candidate: makeCatalogCandidate({
          targetId,
          configurationId: ModelServingConfigurationIdSchema.make(`configuration_${index}`),
        }),
        intent,
        explanation: `${intent} explanation`,
      })
    }
    const intents = ["fastest", "lightweight", "best_quality", "balanced"] as const
    const models = intents.map((intent, index) => makeModel({
      targetId: ModelOfferingTargetIdSchema.make(`target_${index}`),
      displayName: `${intent} model`,
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
    }))
    const selections = selectionsFor(makeView({
      ready: false,
      models,
      recommendations: intents.map(recommendation),
    }))
    expect(selections.map(({ recommendation }) => recommendation._tag === "Recommended"
      ? recommendation.value.intent
      : "none")).toEqual([
      "balanced",
      "best_quality",
      "fastest",
      "lightweight",
    ])
  })

  it("exposes an assessed capacity failure", () => {
    const recommendation = makeRecommendation({
      candidate: makeCatalogCandidate({
        availability: {
          _tag: "Unavailable",
          failure: {
            code: "insufficient_resources",
            message: "This configuration does not fit",
            retryable: true,
          },
        },
      }),
    })
    const selection = selectionsFor(makeView({
      models: [makeModel({
        offerings: [],
        download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 1 },
      })],
      recommendations: [recommendation],
    }))[0]!
    expect(selectionCapacityWarning(selection)).toBe("This configuration does not fit")
  })
})
