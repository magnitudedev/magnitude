import { Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import {
  AssessmentEnvironmentIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelSchema,
  ModelIdSchema,
  ModelAssessmentIdSchema,
  ModelVariantLabelSchema,
  type LocalModel,
  type LocalModelsState,
} from "@magnitudedev/acn-protocol"
import { localProviderOfferingsReady, projectLocalProviderOfferings } from "./local-provider-offerings"

const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
} as const

const assessed = (modelId: string, source: "Catalog" | "Discovered"): LocalModel => {
  const providerModelId = ModelIdSchema.make(modelId)
  const installation = {
    _tag: "Resolved",
    installedBytes: 1_000,
    primaryPath: "/models/model.gguf",
    ownership: source === "Catalog" ? "Magnitude" : "ExternalHuggingFace",
  } as const
  const servingState = {
    _tag: "Assessed",
    metadata: {
      format: "gguf",
      architecture: "test",
      quantization: "Q4",
      quantizationName: "4-bit",
      storageBytes: 1_000,
      maximumContextLength: Option.some(32_768),
    },
    capabilities,
    speculativeMethod: Option.none(),
    assessment: {
      _tag: "Fits",
      assessmentId: ModelAssessmentIdSchema.make(`assessment-${source}`),
      environmentId: AssessmentEnvironmentIdSchema.make("environment"),
      profile: { contextLength: 32_768 },
      memory: {
        domains: [],
        totalRequiredBytes: 0,
        requiredSystemMemoryBytes: 0,
        systemUseState: { _tag: "NotObserved" },
        currentHeadroomState: { _tag: "NotObserved" },
      },
      performance: [{
        contextTokens: 32_768,
        lowerTokensPerSecond: 20,
        estimatedTokensPerSecond: 25,
        upperTokensPerSecond: 30,
        confidence: "high",
      }],
    },
  } as const
  return Schema.validateSync(LocalModelSchema)({
    _tag: source,
    modelId: providerModelId,
    ...(source === "Catalog" ? { storageBytes: 1_000 } : {}),
    presentation: {
      displayName: "Model",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "",
      license: Option.none(),
      sourceUrls: [],
    },
    ...(source === "Catalog"
      ? { catalogData: {
          releaseDate: "2026-08-29",
          parameterization: { architecture: "dense", totalParameters: 1 },
          intelligence: { score: 1, provenance: {
            kind: "artificialAnalysisIntelligenceIndex",
            methodologyVersion: "test",
            asOfDate: "2026-08-29",
            url: "https://example.com/model",
          } },
          fidelityRank: 1,
          quantizationAware: false,
        } }
      : {}),
    ...(source === "Catalog"
      ? { acquisitionState: { _tag: "Installed", installation, residencyState: { _tag: "Unloaded" } } }
      : { state: { _tag: "Ready", installation, residencyState: { _tag: "Unloaded" },
          catalogAttribution: { _tag: "NotInCatalog" }, servingState } }),
    ...(source === "Catalog" ? { servingState: { ...servingState, rankingScores: Option.none() } } : {}),
  })
}

const catalogAssessed = (modelId: string): Extract<LocalModel, { readonly _tag: "Catalog" }> => {
  const model = assessed(modelId, "Catalog")
  if (model._tag !== "Catalog") throw new Error("expected catalog model")
  return model
}

describe("local provider offerings", () => {
  it("preserves canonical catalog and discovered model IDs unchanged", () => {
    const catalog = assessed("qwen3.5-4b:gguf:q4", "Catalog")
    const discovered = assessed("hf:owner/repository/model-q5.gguf", "Discovered")
    const projection = projectLocalProviderOfferings([catalog, discovered])

    expect(projection.offerings.map(({ providerModelId }) => providerModelId)).toEqual([
      catalog.modelId,
      discovered.modelId,
    ])
    expect(projection.offerings.every(({ profile }) => profile.contextLength === 32_768)).toBe(true)
    expect(projection.offerings.every(({ capabilities }) => capabilities.tools)).toBe(true)
  })

  it("does not fabricate offerings or provider metadata before assessment", () => {
    const model = catalogAssessed("qwen3.5-4b:gguf:q4")
    const projection = projectLocalProviderOfferings([{
      ...model,
      servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } },
    }])
    expect(projection).toEqual({ offerings: [], entries: [] })
  })

  it("preserves resource and compatibility reasons in provider catalog entries", () => {
    const model = catalogAssessed("qwen3.5-4b:gguf:q4")
    if (model.servingState._tag !== "Assessed") throw new Error("expected assessed model")
    const serving = model.servingState
    const doesNotFit: LocalModel = {
      ...model,
      servingState: {
        _tag: "Assessed",
        metadata: serving.metadata,
        capabilities: serving.capabilities,
        speculativeMethod: serving.speculativeMethod,
        assessment: {
          _tag: "DoesNotFit",
          assessmentId: ModelAssessmentIdSchema.make("does-not-fit"),
          environmentId: AssessmentEnvironmentIdSchema.make("environment"),
          profile: { contextLength: 32_768 },
          memoryDomains: [{
            memoryDomainId: LocalInferenceMemoryDomainIdSchema.make("system"),
            capacityBytes: 0,
            requiredBytes: 1,
            compatibilityReserveBytes: 0,
            remainingBytes: -1,
          }],
          totalRequiredBytes: 1,
          deficitBytes: 1,
          limitingResource: "system",
        },
      },
    }
    expect(projectLocalProviderOfferings([doesNotFit]).entries[0]?.availability).toEqual({
      _tag: "Disabled",
      reason: "insufficient_resources",
    })
  })

  it("does not make temporary startup or assessment emptiness authoritative", () => {
    const model = catalogAssessed("qwen3.5-4b:gguf:q4")
    const state = (overrides: Partial<LocalModelsState> = {}): LocalModelsState => ({
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: true, settledModels: 1, totalModels: 1 },
      },
      models: [model],
      ...overrides,
    })
    expect(localProviderOfferingsReady(state())).toBe(true)
    expect(localProviderOfferingsReady(state({
      models: [{ ...model, servingState: { _tag: "Assessing", profile: { contextLength: 32_768 } } }],
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: false, settledModels: 0, totalModels: 1 },
      },
    }))).toBe(false)
    expect(localProviderOfferingsReady(state({
      preparation: {
        discovery: { complete: true, modelsFound: 1 },
        assessment: { complete: false, settledModels: 0, totalModels: 1 },
      },
    }))).toBe(false)
  })
})
