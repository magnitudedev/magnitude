import { Option, Schema } from "effect"
import { AssessmentEnvironmentIdSchema, CatalogFormModelIdSchema, LocalInferenceDeviceIdSchema, CustomFormModelIdSchema, LocalModelFitResidencySchema, ModelAssessmentIdSchema, ModelReleaseDateSchema, ModelStreamingStatusSchema, ModelVariantLabelSchema, type AvailableCatalogLocalModel, type CustomLocalModel, type LocalModel, type LocalModelFitResidency, type ModelInstanceAllocation, type ModelResidency, type ModelStreamingStatus } from "@magnitudedev/sdk"
export const providerModelId = CatalogFormModelIdSchema.make("setup-model:gguf:q4")
export const makeSetupModel = (installed: boolean): AvailableCatalogLocalModel => {
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
        aliases: [],
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
    missingComponents: [],
    storedComponents: [],
    drafterWithheld: false,
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
        served: { contextLength: 32_768 },
        assessmentId: ModelAssessmentIdSchema.make("setup-assessment"),
        environmentId: AssessmentEnvironmentIdSchema.make("setup-environment"),
        memory: {
          domains: [],
          totalRequiredBytes: 0,
          requiredSystemMemoryBytes: 0,
          breakdown: { weightsBytes: 0, contextBytes: 0, overheadBytes: 0 },
        },
        residency: {
          _tag: "Resident",
          performance: {
            _tag: "Estimated",
            samples: [{ contextTokens: 32_768, estimatedTokensPerSecond: 50 }],
          },
        },
      },
      rankingScores: {
        intelligence: 0.7,
        speed: installed ? Option.none() : Option.some(0.8),
        fidelity: 0.9,
      },
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
  drafterPending: false,
  drafterChoice: { _tag: "Auto" },
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
      served: { contextLength: 32_768 },
      assessmentId: ModelAssessmentIdSchema.make("found-assessment"),
      environmentId: AssessmentEnvironmentIdSchema.make("setup-environment"),
      memory: { domains: [], totalRequiredBytes: 0, requiredSystemMemoryBytes: 0, breakdown: { weightsBytes: 0, contextBytes: 0, overheadBytes: 0 } },
      residency: {
        _tag: "Resident",
        performance: {
          _tag: "Estimated",
          samples: [{ contextTokens: 32_768, estimatedTokensPerSecond: 40 }],
        },
      },
    },
  },
  storageBytes: 2,
  ...overrides,
})

const estimated = (estimatedTokensPerSecond: number) => ({
  _tag: "Estimated",
  samples: [{ contextTokens: 32_768, estimatedTokensPerSecond }],
})

/**
 * A streamed fit as the protocol carries it: 60% of experts in memory at ~20 tok/s, a floor of 1%
 * (the least its steps hold at once) at ~5 tok/s, the rest kept as `coldExperts` says (read from disk unless given).
 */
export const streamedFitResidency = (coldExperts = "disk"): LocalModelFitResidency =>
  Schema.decodeUnknownSync(LocalModelFitResidencySchema)({
    _tag: "Streamed",
    bank: { experts: 100, slots: 60, floorSlots: 1, coldExperts },
    performance: estimated(20.4),
    floorPerformance: estimated(5.4),
  })

/**
 * An expert bank as the protocol carries it: 40% of every expert held (floor 1%, ceiling 60%),
 * 300 MB in the first memory domain and 400 MB in any later one; beside a device's own memory
 * (any domain but host RAM, "system"), a host-RAM tier of 210 MB at half its most, which unified
 * memory never has; 300 MB on disk, an n-gram table of 126 MB at half its most (floor a quarter);
 * and its experts' last resize, from 50% to 40% for a request.
 */
export const streamingStatus = (memoryDomainIds: readonly string[]): ModelStreamingStatus =>
  Schema.decodeUnknownSync(ModelStreamingStatusSchema)({
    experts: 100,
    slots: 40,
    floorSlots: 1,
    ceilingSlots: 60,
    tiers: memoryDomainIds.map((memoryDomainId, index) => ({ memoryDomainId, bytes: (index === 0 ? 300 : 400) * 1000 ** 2 })),
    ...(memoryDomainIds.some((memoryDomainId) => memoryDomainId !== "system")
      ? { host: { memoryDomainId: "system", slots: 30, ceilingSlots: 60, bytes: 210 * 1000 ** 2 } }
      : {}),
    diskBytes: 300 * 1000 ** 2,
    tables: [{ records: 20, floorRecords: 10, ceilingRecords: 40, bytes: 126 * 1000 ** 2 }],
    lastResize: { _tag: "Shrank", cause: "live_claim", to: 40, by: 10, atUnixMs: 1234 },
  })

/** How a load serves when its prompt groups are granted one place of eight. */
export const servingAllocation = {
  promptGroupPlaces: { held: 1, capacity: 8 },
} as const satisfies Pick<ModelInstanceAllocation, "promptGroupPlaces">

export const streamedReadyResidency = (memoryDomainIds: readonly string[]): ModelResidency => ({
  _tag: "Ready",
  allocation: {
    contextWindowTokens: 4096,
    memoryDomains: [],
    ...servingAllocation,
    streaming: Option.some(streamingStatus(memoryDomainIds)),
  },
  device: { deviceId: LocalInferenceDeviceIdSchema.make("metal:0"), backend: "metal" },
})
