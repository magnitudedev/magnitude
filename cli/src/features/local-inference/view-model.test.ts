import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  CatalogCandidateIdSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  RecommendationIdSchema,
} from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  formatModelLoadProgress,
  localInferenceProgressLines,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"
import { GIB, makeCatalogCandidate, makeHardware, makeModel, makeRecommendation, makeView } from "./test-fixtures"

describe("local inference selection view model", () => {
  it("shows native load progress", () => {
    expect(formatModelLoadProgress(42)).toBe("Loading 42%")
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
      },
      {
        id: "assessment",
        status: { _tag: "Running", startedAtMs: 2_000 },
        completedItems: Option.some(8),
        totalItems: Option.some(28),
      },
    ], 4_000)).toEqual([
      {
        id: "hardware",
        state: "completed",
        label: "Detected hardware",
        metadata: " · 1/1 · 1s",
      },
      {
        id: "inventory",
        state: "completed",
        label: "Checked for downloaded models",
        metadata: " · 2/2 · 0.5s",
      },
      {
        id: "assessment",
        state: "running",
        label: "Evaluating models for this machine",
        metadata: " · 8/28 · 2s",
      },
    ])
  })

  it("identifies reused recommendation work without implying a network refresh", () => {
    expect(localInferenceProgressLines([{
      id: "selection",
      status: {
        _tag: "Completed",
        startedAtMs: 1_000,
        durationMs: 0,
        cached: true,
      },
      completedItems: Option.some(4),
      totalItems: Option.some(4),
    }], 1_000)[0]).toEqual({
      id: "selection",
      state: "completed",
      label: "Prepared recommendations",
      metadata: " · 4/4 · cached",
    })
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
  })

  it("classifies the downloaded model selected by a ready slot as running", () => {
    expect(buildLocalInferenceSelections(makeView())[0]?.kind).toBe("running")
  })

  it("keeps recommendations actionable without duplicating target state", () => {
    const model = makeModel({
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
      preparation: { _tag: "NotDownloaded" },
    })
    const selections = buildLocalInferenceSelections(makeView({
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
      const candidateId = CatalogCandidateIdSchema.make(`candidate_${index}`)
      return makeRecommendation({
        id: RecommendationIdSchema.make(`recommendation_${intent}`),
        candidate: makeCatalogCandidate({ id: candidateId }),
        intent,
        explanation: `${intent} explanation`,
      })
    }
    const intents = ["fastest", "lightweight", "best_quality", "balanced"] as const
    const models = intents.map((intent, index) => makeModel({
      id: ModelOfferingTargetIdSchema.make(`target_${index}`),
      catalogCandidateIds: [CatalogCandidateIdSchema.make(`candidate_${index}`)],
      displayName: `${intent} model`,
      download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
      preparation: { _tag: "NotDownloaded" },
    }))
    const selections = buildLocalInferenceSelections(makeView({
      ready: false,
      models,
      recommendations: intents.map(recommendation),
    }))
    expect(selections.map(({ recommendation: value }) => Option.match(value, {
      onNone: () => "none",
      onSome: ({ intent }) => intent,
    }))).toEqual([
      "balanced",
      "best_quality",
      "fastest",
      "lightweight",
    ])
  })

  it("exposes the target preparation failure", () => {
    const model = makeModel({
      preparation: {
        _tag: "Unavailable",
        providerModelIds: [],
        failure: { code: "does_not_fit", message: "Requires more memory", retryable: false },
      },
    })
    const selection = buildLocalInferenceSelections(makeView({
      models: [model],
      ready: false,
    }))[0]!
    expect(selectionCapacityWarning(selection)).toBe("Requires more memory")
  })
})
