import { Option } from "effect"
import {
  AssessmentEnvironmentIdSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  ModelInstanceIdSchema,
  ModelAssessmentIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogReady,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  RecommendationIdSchema,
  SECONDARY_SLOT_ID,
  type LocalInferenceHardware,
  type LocalModelsState,
  type ModelInstanceAllocation,
  type ModelSlotsState,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelRecommendation,
  type ProviderModelCatalogState,
} from "@magnitudedev/sdk"

export const GIB = 1024 ** 3
export const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export const TEST_MODEL_ID = ProviderModelIdSchema.make("configuration_test")
export const TEST_TARGET_ID = ModelOfferingTargetIdSchema.make("target_test")
export const TEST_CONFIGURATION_ID = ModelServingConfigurationIdSchema.make("configuration_test")
export const TEST_MEMORY_DOMAIN_ID = LocalInferenceMemoryDomainIdSchema.make("memory")
export const TEST_REASONING_EFFORT = ReasoningEffortSchema.make("none")

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
  warningReserveBytes: 13 * GIB,
  assessReserveBytes: 7 * GIB,
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

export const makeModel = (overrides: Partial<LocalModel> = {}): LocalModel => ({
  targetId: TEST_TARGET_ID,
  offerings: [{ configurationId: TEST_CONFIGURATION_ID, providerModelId: TEST_MODEL_ID }],
  displayName: "Qwen Test",
  description: "Test model",
  kind: "Standalone",
  quantization: "Q4_K_M",
  maximumContextLength: 32_768,
  downloadBytes: 16 * GIB,
  download: { _tag: "Downloaded", installedBytes: 16 * GIB },
  assessment: { _tag: "Unassessed" },
  ...overrides,
})

export const makeCatalogCandidate = (
  overrides: Partial<LocalModelCatalogCandidate> = {},
): LocalModelCatalogCandidate => {
  const profile = overrides.profile ?? { contextLength: 32_768 }
  const performanceContexts = [...new Set([
    ...[25_000, 50_000, 75_000].filter((context) =>
      context <= profile.contextLength),
    profile.contextLength,
  ])].sort((left, right) => left - right)
  const performance = overrides.performance ?? performanceContexts.map((contextTokens) => {
    const estimatedTokensPerSecond = contextTokens === profile.contextLength ? 24 : 28
    return {
      contextTokens,
      lowerTokensPerSecond: estimatedTokensPerSecond - 4,
      estimatedTokensPerSecond,
      upperTokensPerSecond: estimatedTokensPerSecond + 4,
      confidence: "moderate" as const,
    }
  })
  return {
    configurationId: TEST_CONFIGURATION_ID,
    assessmentId: ModelAssessmentIdSchema.make("assessment_test"),
    environmentId: AssessmentEnvironmentIdSchema.make("environment_test"),
    targetId: TEST_TARGET_ID,
    displayName: "Qwen Test",
    description: "Test model",
    license: "Apache-2.0",
    profile,
    downloadBytes: 16 * GIB,
    download: { _tag: "NotDownloaded", completedBytes: 0, totalBytes: 16 * GIB },
    availability: { _tag: "NotDownloaded" },
    quantization: "Q4_K_M",
    quantizationName: "4-bit",
    memory: [{
      memoryDomainId: TEST_MEMORY_DOMAIN_ID,
      capacityBytes: 22 * GIB,
      requiredBytes: 18 * GIB,
      compatibilityReserveBytes: 2 * GIB,
      warningReserveBytes: 4 * GIB,
      remainingBytes: 2 * GIB,
    }],
    recommendationEvidence: Option.some({
      intelligence: Option.some({ score: 75, provenance: "Test evidence" }),
      fidelityRank: 75,
      qualityEvidence: ["Test quantization evidence"],
    }),
    performance,
    capabilities: {
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: {
        supported: false,
        efforts: [],
        defaultEffort: Option.none(),
      },
    },
    sources: [],
    ...overrides,
  }
}

export const makeRecommendation = (
  overrides: Partial<LocalModelRecommendation> = {},
): LocalModelRecommendation => ({
  id: RecommendationIdSchema.make("recommendation_test"),
  intent: "balanced",
  explanation: "Balanced local inference.",
  candidate: makeCatalogCandidate(),
  ...overrides,
})

export const makeView = (options: {
  readonly hardware?: LocalInferenceHardware
  readonly models?: readonly LocalModel[]
  readonly recommendations?: readonly LocalModelRecommendation[]
  readonly catalogCandidates?: readonly LocalModelCatalogCandidate[]
  readonly providerContextWindow?: number
  readonly allocation?: ModelInstanceAllocation
  readonly ready?: boolean
} = {}): {
  readonly hardware: LocalInferenceHardware
  readonly models: LocalModelsState
  readonly catalog: ProviderModelCatalogState
  readonly slots: ModelSlotsState
  readonly providerModelId: Option.Option<typeof TEST_MODEL_ID>
} => {
  const models = options.models ?? [makeModel()]
  const selection = {
    providerId: LOCAL_PROVIDER_ID,
    providerModelId: TEST_MODEL_ID,
    reasoningEffort: TEST_REASONING_EFFORT,
  }
  return {
    hardware: options.hardware ?? makeHardware(),
    models: {
      models,
      recommendations: {
        _tag: "Ready",
        entries: options.recommendations ?? [],
        catalog: options.catalogCandidates ?? [],
        progress: [],
      },
    },
    catalog: new ProviderModelCatalogReady({
      providers: [{
        providerId: LOCAL_PROVIDER_ID,
        displayName: "Local",
        kind: "Local",
        authentication: "NotRequired",
        availability: { _tag: "Available" },
      }],
      models: [{
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        modelFamilyId: Option.none(),
        displayName: "Qwen Test",
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: options.providerContextWindow ?? 32_768,
        maxOutputTokens: 4_096,
        memory: Option.none(),
        capabilities: {
          vision: false,
          tools: true,
          structuredOutput: true,
          reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
        },
        availability: { _tag: "Available" },
        pricing: Option.none(),
      }],
    }),
    slots: {
      slots: {
        primary: options.ready === false
          ? new ModelSlotUnassigned({ slotId: PRIMARY_SLOT_ID })
          : new ModelSlotConfiguredLocal({
              slotId: PRIMARY_SLOT_ID,
              selection,
              descriptor: {
                providerId: LOCAL_PROVIDER_ID,
                providerModelId: TEST_MODEL_ID,
                displayName: "Qwen Test",
              },
              availability: { _tag: "Available" },
              instance: Option.some({
                id: ModelInstanceIdSchema.make("test-instance"),
                configurationId: TEST_CONFIGURATION_ID,
                lifecycle: {
                  _tag: "Ready",
                  allocation: options.allocation ?? {
                    contextWindowTokens: 32_768,
                    parallelSequences: 1,
                    physicalContextTokens: 32_768,
                    memoryDomains: [],
                  },
                },
              }),
              actions: ["Stop"],
            }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModelIds: { primary: [TEST_MODEL_ID], secondary: [] },
      favoriteModels: [],
    },
    providerModelId: Option.some(TEST_MODEL_ID),
  }
}
