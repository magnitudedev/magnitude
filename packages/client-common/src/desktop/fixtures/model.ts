import { Option, Schema } from "effect"
import { AssessmentEnvironmentIdSchema, CatalogIntelligenceSchema, CatalogFormModelIdSchema, ModelAssessmentIdSchema, ModelReleaseDateSchema, ModelVariantLabelSchema, type LocalModel } from "@magnitudedev/sdk"
export const providerModelId = CatalogFormModelIdSchema.make("setup-model:gguf:q4")
export const makeSetupModel = (installed: boolean): Extract<LocalModel, { readonly _tag: "Catalog" }> => {
  return {
    _tag: "Catalog",
    modelId: providerModelId,
    storageBytes: 1,
    presentation: {
      displayName: "Setup Model",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "",
      license: Option.none(),
      sourceUrls: [],
    },
    catalogData: {
        releaseDate: ModelReleaseDateSchema.make("2026-01-01"),
        parameterization: { architecture: "dense", totalParameters: 1 },
        intelligence: Schema.decodeUnknownSync(CatalogIntelligenceSchema)({
          score: 1,
          provenance: {
            kind: "artificialAnalysisIntelligenceIndex",
            methodologyVersion: "test",
            asOfDate: "2026-01-01",
            url: "https://example.com/model",
          },
        }),
        fidelityRank: 1,
        quantizationAware: false,
    },
    acquisitionState: installed
      ? {
          _tag: "Installed",
          installation: {
            _tag: "Resolved",
            installedBytes: 1,
            primaryPath: "/models/setup.gguf",
            ownership: "Magnitude",
          },
          residencyState: { _tag: "Unloaded" },
        }
      : { _tag: "NotInstalled" },
    servingState: {
      _tag: "Assessed",
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
      },
      speculativeMethod: Option.none(),
      metadata: {
        format: "gguf",
        quantization: "Q4_K_M",
        quantizationName: "4-bit",
        architecture: "test",
        maximumContextLength: Option.some(32_768),
        storageBytes: 1,
      },
      assessment: {
        _tag: "Fits",
        profile: { contextLength: 32_768 },
        assessmentId: ModelAssessmentIdSchema.make("setup-assessment"),
        environmentId: AssessmentEnvironmentIdSchema.make("setup-environment"),
        memory: {
          domains: [],
          totalRequiredBytes: 0,
          requiredSystemMemoryBytes: 0,
          systemUseState: {
            _tag: "WithinRecommendedHeadroom",
            recommendedHeadroomBytes: 0,
            predictedHeadroomBytes: 0,
          },
          currentHeadroomState: { _tag: "NotObserved" },
        },
        performance: [],
      },
      rankingScores: installed
        ? Option.none()
        : Option.some({ intelligence: 0.7, speed: 0.8, fidelity: 0.9 }),
    },
  }
}
