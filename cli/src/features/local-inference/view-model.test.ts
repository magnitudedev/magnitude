import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  RecommendationIdSchema,
} from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  formatModelLoadProgress,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"
import { GIB, makeHardware, makeModel, makeRecommendation, makeView } from "./test-fixtures"

describe("local inference selection view model", () => {
  it("shows native load progress", () => {
    expect(formatModelLoadProgress(42)).toBe("Loading 42%")
  })

  it("presents unified memory from the hardware contract", () => {
    const memoryDomainId = LocalInferenceMemoryDomainIdSchema.make("unified")
    const hardware = makeHardware({
      platform: "MacOS",
      architecture: "Arm64",
      processor: Option.some("Apple M4 Max"),
      totalSystemMemoryBytes: 64 * GIB,
      accelerators: [{
        acceleratorId: LocalInferenceAcceleratorIdSchema.make("metal"),
        name: "Apple M4 Max",
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
    ) => makeRecommendation({
      id: RecommendationIdSchema.make(`recommendation_${intent}`),
      modelId: ModelOfferingTargetIdSchema.make(`target_${index}`),
      intent,
      explanation: `${intent} explanation`,
    })
    const intents = ["fastest", "lightweight", "best_quality", "balanced"] as const
    const models = intents.map((intent, index) => makeModel({
      id: ModelOfferingTargetIdSchema.make(`target_${index}`),
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
