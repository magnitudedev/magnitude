import { Option } from "effect"
import {
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  LocalModelDownloaded,
  LocalModelIdSchema,
  ModelSlotReady,
  ModelSlotUnassigned,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type LocalInferenceHardware,
  type LocalModelInventoryEntry,
  type LocalModelInventoryEntryDetails,
} from "@magnitudedev/sdk"
import type { LocalInferenceView } from "@magnitudedev/client-common"

export const GIB = 1024 ** 3
export const LOCAL_PROVIDER_ID = ProviderIdSchema.make("local")
export const TEST_MODEL_ID = ProviderModelIdSchema.make("local:test-model")
export const TEST_LOCAL_MODEL_ID = LocalModelIdSchema.make("test-model")
export const TEST_MEMORY_DOMAIN_ID = LocalInferenceMemoryDomainIdSchema.make("memory")
export const TEST_REASONING_EFFORT = ReasoningEffortSchema.make("none")

export const makeHardware = (
  overrides: Partial<LocalInferenceHardware> = {},
): LocalInferenceHardware => ({
  platform: "Linux",
  architecture: "X64",
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

export const makeModel = (
  overrides: Partial<LocalModelInventoryEntryDetails> = {},
): LocalModelInventoryEntryDetails => ({
  localModelId: TEST_LOCAL_MODEL_ID,
  providerModelId: TEST_MODEL_ID,
  modelFamilyId: Option.none(),
  displayName: "Qwen Test",
  family: "qwen",
  architecture: "Dense",
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: false, efforts: [], defaultEffort: Option.none() },
  },
  contextWindow: 32_768,
  maxOutputTokens: 4_096,
  quantization: "Q4_K_M",
  downloadBytes: 16 * GIB,
  fit: {
    _tag: "Fits",
    requiredBytes: 18 * GIB,
    availableBytes: 22 * GIB,
    memoryDomainIds: [TEST_MEMORY_DOMAIN_ID],
  },
  recommendation: Option.none(),
  ...overrides,
})

export const makeDownloadedEntry = (
  model = makeModel(),
): LocalModelInventoryEntry => new LocalModelDownloaded({ model, downloadedBytes: model.downloadBytes })

export const makeView = (options: {
  readonly hardware?: LocalInferenceHardware
  readonly entries?: readonly LocalModelInventoryEntry[]
  readonly ready?: boolean
} = {}): LocalInferenceView => {
  const model = options.entries?.[0]?.model ?? makeModel()
  const selection = {
    providerId: LOCAL_PROVIDER_ID,
    providerModelId: model.providerModelId,
    reasoningEffort: TEST_REASONING_EFFORT,
  }
  return {
    hardware: options.hardware ?? makeHardware(),
    inventory: { _tag: "Ready", entries: options.entries ?? [makeDownloadedEntry(model)] },
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
