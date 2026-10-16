import { Option, Schema } from "effect"
import { AssessmentEnvironmentIdSchema, CatalogFormModelIdSchema, CustomFormModelIdSchema, ModelAssessmentIdSchema, ModelReleaseDateSchema, ModelVariantLabelSchema, type CustomLocalModel, type LocalModel } from "@magnitudedev/sdk"
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
        intelligence: 1,
        support: { _tag: "Supported" },
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
        },
        performance: [{ contextTokens: 32_768, estimatedTokensPerSecond: 50 }],
      },
      rankingScores: installed
        ? Option.none()
        : Option.some({ intelligence: 0.7, speed: 0.8, fidelity: 0.9 }),
    },
  }
}

export const customModelId = CustomFormModelIdSchema.make("hf:owner/Setup-Model-Uncensored-GGUF/Setup-Model-Q4_K_P.gguf")
export const makeFoundModel = (
  overrides: Partial<CustomLocalModel> = {},
): CustomLocalModel => ({
  _tag: "Custom",
  modelId: customModelId,
  origin: { _tag: "Found", source: { _tag: "HuggingFaceCache" } },
  presentation: {
    displayName: "Setup Model Uncensored",
    variantLabel: ModelVariantLabelSchema.make("Q4_K_P"),
    sourceLabel: "Hugging Face cache",
    variantOf: Option.some(providerModelId),
    license: Option.none(),
    sourceUrls: [],
  },
  drafter: Option.none(),
  recommendedDrafter: Option.none(),
  acquisitionState: {
    _tag: "External",
    installation: {
      _tag: "Resolved",
      installedBytes: 2,
      primaryPath: "/cache/Setup-Model-Q4_K_P.gguf",
      ownership: "External",
    },
    residencyState: { _tag: "Unloaded" },
  },
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
      quantization: "Q4_K",
      quantizationName: "Q4_K_P",
      architecture: "test",
      maximumContextLength: Option.some(32_768),
      storageBytes: 2,
    },
    assessment: {
      _tag: "Fits",
      profile: { contextLength: 32_768 },
      assessmentId: ModelAssessmentIdSchema.make("found-assessment"),
      environmentId: AssessmentEnvironmentIdSchema.make("setup-environment"),
      memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0 },
      performance: [{ contextTokens: 32_768, estimatedTokensPerSecond: 40 }],
    },
  },
  storageBytes: 2,
  ...overrides,
})
