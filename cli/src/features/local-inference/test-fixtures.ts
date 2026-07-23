import { Option } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  ModelOfferingTargetIdSchema,
  ModelSlotReady,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelCatalogReady,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  RecommendationIdSchema,
  SECONDARY_SLOT_ID,
  type LocalInferenceHardware,
  type LocalModel,
  type LocalModelRecommendation,
} from "@magnitudedev/sdk"
import type { LocalInferenceView } from "@magnitudedev/client-common"

export const GIB = 1024 ** 3
export const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export const TEST_MODEL_ID = ProviderModelIdSchema.make("local:test-model")
export const TEST_TARGET_ID = ModelOfferingTargetIdSchema.make("target_test")
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
  availableSystemMemoryBytes: Option.some(12 * GIB),
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
  residentMemory: Option.none(),
  ...overrides,
})

export const makeModel = (overrides: Partial<LocalModel> = {}): LocalModel => ({
  id: TEST_TARGET_ID,
  displayName: "Qwen Test",
  description: "Test model",
  kind: "Standalone",
  quantization: "Q4_K_M",
  maximumContextLength: 32_768,
  downloadBytes: 16 * GIB,
  download: { _tag: "Downloaded", installedBytes: 16 * GIB },
  preparation: { _tag: "Available", providerModelIds: [TEST_MODEL_ID] },
  ...overrides,
})

export const makeRecommendation = (
  overrides: Partial<LocalModelRecommendation> = {},
): LocalModelRecommendation => ({
  id: RecommendationIdSchema.make("recommendation_test"),
  modelId: TEST_TARGET_ID,
  displayName: "Qwen Test",
  intent: "balanced",
  explanation: "Balanced local inference.",
  sources: [],
  qualityScoreProvenance: "Test evidence",
  fidelityRank: 0,
  qualityEvidence: ["Test quantization evidence"],
  profile: { contextLength: 32_768, parallelSequences: 1 },
  fit: {
    requiredBytes: 18 * GIB,
    availableBytes: 22 * GIB,
    estimatedTokensPerSecond: Option.none(),
  },
  ...overrides,
})

export const makeView = (options: {
  readonly hardware?: LocalInferenceHardware
  readonly models?: readonly LocalModel[]
  readonly recommendations?: readonly LocalModelRecommendation[]
  readonly ready?: boolean
} = {}): LocalInferenceView => {
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
        progress: [],
      },
    },
    catalog: new ProviderModelCatalogReady({
      providers: [{
        providerId: LOCAL_PROVIDER_ID,
        displayName: "Local",
        authentication: "NotRequired",
        availability: { _tag: "Available" },
      }],
      models: [{
        providerId: LOCAL_PROVIDER_ID,
        providerModelId: TEST_MODEL_ID,
        modelFamilyId: Option.none(),
        displayName: "Qwen Test",
        supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
        contextWindow: 32_768,
        maxOutputTokens: 4_096,
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
          : new ModelSlotReady({ slotId: PRIMARY_SLOT_ID, selection }),
        secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
      },
    },
  }
}
