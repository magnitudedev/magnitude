import { Option, Schema } from "effect"
import {
  AssessmentEnvironmentIdSchema,
  CatalogIntelligenceSchema,
  CatalogFormModelIdSchema,
  HuggingFaceFormModelIdSchema,
  HttpsUrlSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelAssessmentIdSchema,
  ModelVariantLabelSchema,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type ModelReleaseDate,
} from "@magnitudedev/sdk"

const GIB = 1024 ** 3
const TEST_MODEL_ID = HuggingFaceFormModelIdSchema.make("hf:test/model/model-q4.gguf")
const TEST_CATALOG_MODEL_ID = CatalogFormModelIdSchema.make("qwen-test:gguf:q4")
const TEST_MEMORY_DOMAIN_ID = LocalInferenceMemoryDomainIdSchema.make("memory")
type DiscoveredLocalModel = Extract<LocalModel, { readonly _tag: "Discovered" }>
type CatalogLocalModel = Extract<LocalModel, { readonly _tag: "Catalog" }>

export const makeHardware = (
  overrides: Partial<LocalInferenceHardware> = {},
): LocalInferenceHardware => ({
  platform: "Linux",
  architecture: "X64",
  productName: Option.none(),
  processor: Option.some("Test CPU"),
  logicalCores: 16,
  totalSystemMemoryBytes: 64 * GIB,
  availableSystemMemoryBytes: 12 * GIB,
  systemAllocationCapacityBytes: 64 * GIB,
  systemAllocationHeadroomBytes: 12 * GIB,
  abortReserveBytes: 4 * GIB,
  accelerators: [{
    acceleratorId: LocalInferenceAcceleratorIdSchema.make("gpu"),
    name: "Test GPU",
    backend: "CUDA",
    memoryDomainId: TEST_MEMORY_DOMAIN_ID,
  }],
  memoryDomains: [{
    memoryDomainId: TEST_MEMORY_DOMAIN_ID,
    kind: "PhysicalDevice",
    totalBytes: 24 * GIB,
    stableCapacityBytes: 22 * GIB,
    availableBytes: Option.some(6 * GIB),
    sharesSystemMemory: false,
  }],
  ...overrides,
})

const capabilities = {
  vision: false,
  tools: true,
  structuredOutput: true,
  reasoning: { supported: false as const, efforts: [], defaultEffort: Option.none() },
}

const performance = (contextLength: number) => [...new Set([
  ...[25_000, 50_000, 75_000].filter((context) => context <= contextLength),
  contextLength,
])].sort((left, right) => left - right).map((contextTokens) => {
  const estimatedTokensPerSecond = contextTokens === contextLength ? 24 : 28
  return {
    contextTokens,
    lowerTokensPerSecond: estimatedTokensPerSecond - 4,
    estimatedTokensPerSecond,
    upperTokensPerSecond: estimatedTokensPerSecond + 4,
    confidence: "moderate" as const,
  }
})

type ReadyDiscoveredLocalModel = Omit<DiscoveredLocalModel, "state"> & {
  readonly state: Extract<DiscoveredLocalModel["state"], { readonly _tag: "Ready" }>
}

const makeModel = (overrides: Partial<ReadyDiscoveredLocalModel> = {}): ReadyDiscoveredLocalModel => {
  const contextLength = 32_768
  return {
    _tag: "Discovered",
    modelId: overrides.modelId ?? TEST_MODEL_ID,
    presentation: {
      displayName: "Qwen Test",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "Test model",
      license: Option.none(),
      sourceUrls: [HttpsUrlSchema.make("https://huggingface.co/test/model")],
    },
    state: {
      _tag: "Ready",
      installation: {
        _tag: "Resolved",
        installedBytes: 16 * GIB,
        primaryPath: "/models/model-q4.gguf",
        ownership: "ExternalHuggingFace",
      },
      residencyState: { _tag: "Unloaded" },
      catalogAttribution: { _tag: "NotInCatalog" },
      servingState: {
        _tag: "Assessed",
        metadata: {
          format: "gguf",
          quantization: "Q4_K_M",
          quantizationName: "4-bit",
          architecture: "test",
          storageBytes: 16 * GIB,
          maximumContextLength: Option.some(contextLength),
        },
        capabilities,
        speculativeMethod: Option.none(),
        assessment: {
          _tag: "Fits",
          profile: { contextLength },
          assessmentId: ModelAssessmentIdSchema.make("assessment_test"),
          environmentId: AssessmentEnvironmentIdSchema.make("environment_test"),
          memory: {
            domains: [],
            totalRequiredBytes: 0,
            requiredSystemMemoryBytes: 0,
            systemUseState: {
              _tag: "WithinRecommendedHeadroom",
              recommendedHeadroomBytes: 4 * GIB,
              predictedHeadroomBytes: 48 * GIB,
            },
            currentHeadroomState: { _tag: "NotObserved" },
          },
          performance: performance(contextLength),
        },
      },
    },
    ...overrides,
  }
}

const makeCatalogOnlyModel = (
  overrides: Partial<CatalogLocalModel> = {},
  modelId = TEST_CATALOG_MODEL_ID,
): CatalogLocalModel => {
  const model = makeModel()
  const { state: _state, ...shared } = model
  const servingState = model.state.servingState
  if (servingState._tag !== "Assessed" || servingState.assessment._tag !== "Fits") {
    throw new Error("Base fixture must have a fitting assessment")
  }
  return {
    ...shared,
    _tag: "Catalog",
    modelId,
    storageBytes: servingState.metadata.storageBytes,
    catalogData: {
        releaseDate: "2026-01-01" as ModelReleaseDate,
        parameterization: { architecture: "dense", totalParameters: 8_000_000_000 },
        intelligence: Schema.decodeUnknownSync(CatalogIntelligenceSchema)({
          score: 75,
          provenance: {
            kind: "artificialAnalysisIntelligenceIndex",
            methodologyVersion: "test",
            asOfDate: "2026-01-01",
            url: "https://example.com/model",
          },
        }),
        fidelityRank: 75,
        quantizationAware: false,
    },
    acquisitionState: { _tag: "NotInstalled" },
    servingState: {
      _tag: "Assessed",
      metadata: servingState.metadata,
      capabilities: servingState.capabilities,
      speculativeMethod: servingState.speculativeMethod,
      assessment: servingState.assessment,
      rankingScores: Option.some({ intelligence: 0.75, speed: 0.65, fidelity: 0.75 }),
    },
    presentation: { ...model.presentation, license: Option.some("Apache-2.0") },
    ...overrides,
  }
}

export const makeCatalogModel = (overrides: Partial<CatalogLocalModel> = {}): CatalogLocalModel =>
  makeCatalogOnlyModel(overrides)

export const makeInstalledCatalogModel = (
  overrides: Partial<CatalogLocalModel> = {},
): CatalogLocalModel => {
  const model = makeCatalogOnlyModel()
  if (model.servingState._tag !== "Assessed") throw new Error("Base fixture must be assessed")
  return {
    ...model,
    acquisitionState: {
      _tag: "Installed",
      installation: {
        _tag: "Resolved",
        installedBytes: model.servingState.metadata.storageBytes,
        primaryPath: "/models/catalog-model.gguf",
        ownership: "Magnitude",
      },
      residencyState: { _tag: "Unloaded" },
    },
    ...overrides,
  }
}

export const makeAcquiringModel = (
  acquisitionState: LocalModelAcquisitionState,
  overrides: Partial<CatalogLocalModel> = {},
): CatalogLocalModel => makeCatalogOnlyModel({ acquisitionState, ...overrides })
