import { Option } from "effect"
import {
  AssessmentEnvironmentIdSchema,
  CatalogModelIdSchema,
  CatalogVariantIdSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelPackageIdSchema,
  ModelServingConfigurationIdSchema,
  ModelInstanceIdSchema,
  ModelAssessmentIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotUnassigned,
  ModelVariantLabelSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogReady,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  RecommendationIdSchema,
  SECONDARY_SLOT_ID,
  servableModelBundlePackages,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelAcquisitionState,
  type LocalModelRecommendation,
  type LocalModelsState,
  type ModelInstanceAllocation,
  type ModelSlotsState,
  type ProviderModelCatalogState,
  type ServableModelBundle,
} from "@magnitudedev/sdk"

export const GIB = 1024 ** 3
export const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export const TEST_MODEL_ID = ProviderModelIdSchema.make("configuration_test")
export const TEST_PACKAGE_ID = ModelPackageIdSchema.make("package_test")
export const TEST_CONFIGURATION_ID = ModelServingConfigurationIdSchema.make("configuration_test")
export const TEST_MEMORY_DOMAIN_ID = LocalInferenceMemoryDomainIdSchema.make("memory")
export const TEST_REASONING_EFFORT = ReasoningEffortSchema.make("none")

export const makeStandaloneBundle = (id: string = TEST_PACKAGE_ID): ServableModelBundle => ({
  _tag: "Standalone",
  package: {
    id: ModelPackageIdSchema.make(id),
    source: { _tag: "Local", path: `/models/${id}` },
    files: [],
    relationships: [],
    properties: {
      format: "gguf",
      quantization: "Q4_K_M",
      quantizationName: "4-bit",
      architecture: "test",
      maximumContextLength: Option.some(32_768),
      intrinsicModelId: Option.none(),
      intrinsicQualityId: Option.none(),
    },
  },
})

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

export const makeRecommendation = (
  overrides: Partial<LocalModelRecommendation> = {},
): LocalModelRecommendation => ({
  id: RecommendationIdSchema.make("recommendation_test"),
  intent: "balanced",
  explanation: "Balanced local inference.",
  ...overrides,
})

export const makeModel = (overrides: Partial<LocalModel> = {}): LocalModel => {
  const bundle = overrides.bundle ?? makeStandaloneBundle()
  const contextLength = 32_768
  const [firstInstalledPackage, ...remainingInstalledPackages] = servableModelBundlePackages(bundle)
    .map((modelPackage) => ({
      packageId: modelPackage.id,
      path: modelPackage.source._tag === "Local"
        ? modelPackage.source.path
        : `/models/${modelPackage.id}`,
      origin: "Magnitude" as const,
    }))
  if (firstInstalledPackage === undefined) throw new Error("Test model bundle must contain a package")
  return {
    bundle,
    presentation: {
      displayName: "Qwen Test",
      variantLabel: ModelVariantLabelSchema.make("Q4"),
      description: "Test model",
      license: Option.none(),
    },
    downloadBytes: 16 * GIB,
    catalogMembershipState: { _tag: "NotInCatalog" },
    acquisitionState: {
      _tag: "Installed",
      installedBytes: 16 * GIB,
      packages: [firstInstalledPackage, ...remainingInstalledPackages],
    },
    upgradeState: { _tag: "NotApplicable" },
    servingState: {
      _tag: "Assessed",
      capabilities,
      configuration: {
        id: TEST_CONFIGURATION_ID,
        bundle,
        profile: { contextLength },
      },
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
      availabilityState: { _tag: "Selectable", providerModelId: TEST_MODEL_ID },
      recommendations: [],
    },
    ...overrides,
  }
}

export const makeCatalogOnlyModel = (
  overrides: Partial<LocalModel> = {},
  configurationId = TEST_CONFIGURATION_ID,
): LocalModel => {
  const model = makeModel()
  if (model.servingState._tag !== "Assessed") return { ...model, ...overrides }
  return {
    ...model,
    catalogMembershipState: {
      _tag: "InCatalog",
      catalogData: {
        modelId: CatalogModelIdSchema.make("qwen-test"),
        variantId: CatalogVariantIdSchema.make("gguf:q4"),
        parameterization: { architecture: "dense", totalParameters: 8_000_000_000 },
        intelligenceScore: 75,
        intelligenceScoreSource: "Test catalog score",
        fidelityRank: 75,
        quantizationAware: false,
        qualityNotes: ["Test quantization notes"],
      },
    },
    acquisitionState: { _tag: "NotInstalled", completedBytes: 0, totalBytes: model.downloadBytes },
    upgradeState: { _tag: "NotApplicable" },
    servingState: {
      ...model.servingState,
      configuration: { ...model.servingState.configuration, id: configurationId },
      availabilityState: { _tag: "Installable" },
    },
    presentation: { ...model.presentation, license: Option.some("Apache-2.0") },
    ...overrides,
  }
}

export const withDoesNotFitAssessment = (model: LocalModel): LocalModel => {
  if (model.servingState._tag !== "Assessed"
    || model.servingState.assessment._tag !== "Fits") {
    throw new Error("DoesNotFit fixture requires a fitting assessed model")
  }
  return {
    ...model,
    servingState: {
      ...model.servingState,
      assessment: {
        _tag: "DoesNotFit",
        assessmentId: model.servingState.assessment.assessmentId,
        environmentId: model.servingState.assessment.environmentId,
        memoryDomains: [],
        totalRequiredBytes: 10,
        deficitBytes: 2,
        limitingResource: "system memory",
      },
      availabilityState: {
        _tag: "Unavailable",
        providerModelId: Option.none(),
        failure: {
          code: "insufficient_resources",
          message: "Does not fit",
          retryable: false,
        },
      },
    },
  }
}

export const makeConfiguredModel = (
  configurationId: ReturnType<typeof ModelServingConfigurationIdSchema.make>,
  overrides: Partial<LocalModel> = {},
): LocalModel => {
  const model = makeModel()
  if (model.servingState._tag !== "Assessed") return { ...model, ...overrides }
  return {
    ...model,
    servingState: {
      ...model.servingState,
      configuration: { ...model.servingState.configuration, id: configurationId },
      availabilityState: {
        _tag: "Selectable",
        providerModelId: ProviderModelIdSchema.make(configurationId),
      },
    },
    ...overrides,
  }
}

export const makeModelWithContext = (
  contextLength: number,
  overrides: Partial<LocalModel> = {},
): LocalModel => {
  const model = makeModel()
  if (model.servingState._tag !== "Assessed") return { ...model, ...overrides }
  return {
    ...model,
    servingState: {
      ...model.servingState,
      configuration: {
        ...model.servingState.configuration,
        profile: { contextLength },
      },
      assessment: model.servingState.assessment._tag === "Fits"
        ? {
            ...model.servingState.assessment,
            profile: { contextLength },
            performance: performance(contextLength),
          }
        : model.servingState.assessment,
    },
    ...overrides,
  }
}

export const makeCatalogModel = (overrides: Partial<LocalModel> = {}): LocalModel =>
  makeCatalogOnlyModel(overrides)

export const makeAcquiringModel = (
  acquisitionState: LocalModelAcquisitionState,
  overrides: Partial<LocalModel> = {},
): LocalModel => makeCatalogOnlyModel({ acquisitionState, ...overrides })

export const makeView = (options: {
  readonly hardware?: LocalInferenceHardware
  readonly models?: readonly LocalModel[]
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
      inventoryState: { _tag: "Ready" },
      models,
      discoveryState: { _tag: "Ready", progress: [] },
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
        variantLabel: Option.some(ModelVariantLabelSchema.make("Q4")),
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: options.providerContextWindow ?? 32_768,
        maxOutputTokens: 4_096,
        memory: Option.none(),
        capabilities,
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
                variantLabel: Option.some(ModelVariantLabelSchema.make("Q4")),
              },
              availability: { _tag: "Available" },
              residency: {
                _tag: "Ready",
                instanceId: ModelInstanceIdSchema.make("test-instance"),
                configurationId: TEST_CONFIGURATION_ID,
                allocation: options.allocation ?? {
                  contextWindowTokens: 32_768,
                  parallelSequences: 1,
                  physicalContextTokens: 32_768,
                  memoryDomains: [],
                },
              },
              actions: ["Stop"],
            }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
      recentModels: {
        primary: [{ providerId: LOCAL_PROVIDER_ID, providerModelId: TEST_MODEL_ID }],
        secondary: [],
      },
      favoriteModels: [],
    },
    providerModelId: Option.some(TEST_MODEL_ID),
  }
}
