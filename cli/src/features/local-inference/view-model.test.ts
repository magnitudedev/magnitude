import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  CatalogCandidateIdSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ProviderModelCatalogLoading,
  ProviderModelCatalogReady,
  ProviderModelCatalogRefreshing,
  ProviderModelIdSchema,
  RecommendationIdSchema,
} from "@magnitudedev/sdk"
import {
  buildLocalInferenceSelections,
  describeLocalHardware,
  describeResidentModel,
  localInferenceSetupPhase,
  localInferenceProgressLines,
  selectionCapacityWarning,
  selectionMetadata,
} from "./view-model"
import { GIB, makeCatalogCandidate, makeHardware, makeModel, makeRecommendation, makeView } from "./test-fixtures"

describe("local inference onboarding presentation", () => {
  const loadingRecommendations = {
    _tag: "Loading" as const,
    progress: [],
  }

  it("keeps onboarding in discovery while recommendations are loading", () => {
    const view = makeView()
    expect(localInferenceSetupPhase({
      ...view,
      models: { ...view.models, recommendations: loadingRecommendations },
    })).toBe("discovering")
  })

  it("keeps onboarding in discovery until the initial provider catalog settles", () => {
    const view = makeView()
    expect(localInferenceSetupPhase({
      ...view,
      catalog: new ProviderModelCatalogLoading(),
    })).toBe("discovering")
  })

  it("uses retained catalog data during a refresh", () => {
    const view = makeView()
    if (view.catalog._tag !== "Ready") throw new Error("expected ready catalog fixture")
    expect(localInferenceSetupPhase({
      ...view,
      catalog: new ProviderModelCatalogRefreshing({
        providers: view.catalog.providers,
        models: view.catalog.models,
        failures: [],
      }),
    })).toBe("ready")
  })

  it("classifies a terminal recommendation failure separately from discovery", () => {
    const view = makeView()
    expect(localInferenceSetupPhase({
      ...view,
      models: {
        ...view.models,
        recommendations: {
          _tag: "Failed",
          failure: {
            code: "recommendations_unavailable",
            message: "Recommendation setup failed",
            retryable: true,
          },
          progress: [],
        },
      },
    })).toBe("failed")
  })
})

describe("local inference selection view model", () => {
  it("presents the exact resident model allocation", () => {
    const view = makeView({
      hardware: makeHardware({
        residentRuntime: Option.some({
          configurationId: "configuration_test",
          contextWindowTokens: 200_000,
          parallelSequences: 4,
          physicalContextTokens: 800_000,
        }),
      }),
    })

    expect(describeResidentModel(view.hardware, Option.some(view.catalog))).toEqual(
      Option.some({
        displayName: "Qwen Test",
        contextWindowTokens: 200_000,
        parallelSequences: 4,
      }),
    )
  })

  it("labels the resident configuration rather than the first local catalog model", () => {
    const view = makeView()
    if (view.catalog._tag !== "Ready") throw new Error("expected ready catalog fixture")
    const residentConfigurationId = ModelServingConfigurationIdSchema.make("resident")
    const residentProviderModelId = ProviderModelIdSchema.make("local:resident")
    const otherModel = {
      ...view.catalog.models[0]!,
      providerModelId: ProviderModelIdSchema.make("local:other"),
      displayName: "Other Local Model",
    }
    const residentModel = {
      ...view.catalog.models[0]!,
      providerModelId: residentProviderModelId,
      displayName: "Resident Local Model",
    }
    const catalog = new ProviderModelCatalogReady({
      providers: view.catalog.providers,
      models: [otherModel, residentModel],
    })
    const hardware = makeHardware({
      residentRuntime: Option.some({
        configurationId: residentConfigurationId,
        contextWindowTokens: 64_000,
        parallelSequences: 2,
        physicalContextTokens: 128_000,
      }),
    })

    expect(describeResidentModel(hardware, Option.some(catalog))).toEqual(
      Option.some({
        displayName: "Resident Local Model",
        contextWindowTokens: 64_000,
        parallelSequences: 2,
      }),
    )
  })

  it("retains resident state while catalog metadata is unavailable", () => {
    const hardware = makeHardware({
      residentRuntime: Option.some({
        configurationId: ModelServingConfigurationIdSchema.make("resident"),
        contextWindowTokens: 64_000,
        parallelSequences: 2,
        physicalContextTokens: 128_000,
      }),
    })

    expect(describeResidentModel(hardware, Option.none())).toEqual(
      Option.some({
        displayName: "local:resident",
        contextWindowTokens: 64_000,
        parallelSequences: 2,
      }),
    )
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
        id: "analysis",
        status: { _tag: "Running", startedAtMs: 2_000 },
        completedItems: Option.some(8),
        totalItems: Option.some(28),
        estimatedRemainingMs: Option.some(5_000),
      },
      {
        id: "analysis",
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
        id: "analysis",
        state: "running",
        label: "Evaluating models for this machine",
        metadata: " · 8/28 · about 5s left",
      },
      {
        id: "analysis",
        state: "completed",
        label: "Evaluated 20 models for this machine",
        metadata: " · 8s",
      },
    ])
  })

  it("keeps cache reuse and recommendation timing out of presentation", () => {
    expect(localInferenceProgressLines([
      {
        id: "analysis",
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
        id: "analysis",
        state: "completed",
        label: "Evaluated 20 models for this machine",
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
