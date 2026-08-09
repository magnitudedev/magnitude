import { describe, expect, it } from "vitest"
import { Option } from "effect"
import {
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type LocalModel,
  type LocalModelCatalogCandidate,
  type LocalModelsState,
  type ModelSlotsState,
  type ProviderModelCatalogState,
} from "@magnitudedev/sdk"
import { buildLocalInferenceSelections } from "./local-inference-selections"

const targetId = ModelOfferingTargetIdSchema.make("target")
const configurationId = ModelServingConfigurationIdSchema.make("configuration")
const providerModelId = ProviderModelIdSchema.make("provider-model")
const reasoningEffort = ReasoningEffortSchema.make("none")

const model = (offerings: LocalModel["offerings"] = []): LocalModel => ({
  targetId,
  offerings,
  displayName: "Test model",
  description: "A local test model",
  kind: "Standalone",
  quantization: "Q4",
  maximumContextLength: 100_000,
  downloadBytes: 1_000,
  download: { _tag: "Downloaded", installedBytes: 1_000 },
  assessment: { _tag: "Assessed", environmentId: "environment" as never, configurationIds: [configurationId] },
})

const candidate = (): LocalModelCatalogCandidate => ({
  configurationId,
  assessmentId: "assessment" as never,
  environmentId: "environment" as never,
  targetId,
  displayName: "Test model",
  description: "A local test model",
  license: "Apache-2.0",
  profile: { contextLength: 100_000 },
  downloadBytes: 1_000,
  quantization: "Q4",
  quantizationName: "Q4",
  memory: [],
  performance: [{
    contextTokens: 100_000,
    lowerTokensPerSecond: 10,
    estimatedTokensPerSecond: 12,
    upperTokensPerSecond: 14,
    confidence: "high",
  }],
  capabilities: {
    vision: false,
    tools: true,
    structuredOutput: true,
    reasoning: { supported: true, efforts: [reasoningEffort], defaultEffort: Option.some(reasoningEffort) },
  },
  recommendationEvidence: Option.none(),
  sources: [],
  download: { _tag: "Downloaded", installedBytes: 1_000 },
  availability: { _tag: "Available" },
})

const slots = (ready = false): ModelSlotsState => ({
  slots: {
    primary: ready ? {
      _tag: "ConfiguredLocal",
      slotId: PRIMARY_SLOT_ID,
      selection: { providerId: ProviderIdSchema.make("local"), providerModelId, reasoningEffort },
      descriptor: { providerId: ProviderIdSchema.make("local"), providerModelId, displayName: "Test model" },
      availability: { _tag: "Available" },
      instance: Option.some({
        id: "instance" as never,
        configurationId,
        lifecycle: { _tag: "Ready", allocation: { contextWindowTokens: 100_000, parallelSequences: 1, physicalContextTokens: 100_000, memoryDomains: [] } },
      }),
      actions: ["Stop"],
    } : { _tag: "Unassigned", slotId: PRIMARY_SLOT_ID },
    secondary: { _tag: "Unassigned", slotId: SECONDARY_SLOT_ID },
  },
  recentModelIds: { primary: [], secondary: [] },
  favoriteModels: [],
} as ModelSlotsState)

const catalog = (providerId = ProviderIdSchema.make("local")): ProviderModelCatalogState => ({
  _tag: "Ready",
  providers: [{ providerId, displayName: "Provider", authentication: "NotRequired", availability: { _tag: "Available" } }],
  models: [{
    providerId,
    providerModelId,
    modelFamilyId: Option.none(),
    displayName: "Test model",
    supportedSlots: [PRIMARY_SLOT_ID],
    contextWindow: 100_000,
    maxOutputTokens: 4_096,
    memory: Option.none(),
    capabilities: candidate().capabilities,
    availability: { _tag: "Available" },
    pricing: Option.none(),
  }],
} as ProviderModelCatalogState)

const models = (value: LocalModel, withCandidate = true): LocalModelsState => ({
  models: [value],
  recommendations: {
    _tag: "Ready",
    entries: [],
    catalog: withCandidate ? [candidate()] : [],
    progress: [],
  },
})

describe("buildLocalInferenceSelections", () => {
  it("keeps an installed assessed target actionable before an offering exists", () => {
    const result = buildLocalInferenceSelections(models(model()), catalog(), slots())
    expect(result).toHaveLength(1)
    expect(result[0]).toMatchObject({ kind: "stored", configurationId })
    expect(Option.isNone(result[0]!.providerModelId)).toBe(true)
  })

  it("correlates an exact local offering and ready slot as running", () => {
    const result = buildLocalInferenceSelections(
      models(model([{ configurationId, providerModelId }])),
      catalog(),
      slots(true),
    )
    expect(result[0]).toMatchObject({ kind: "running", configurationId })
    expect(Option.getOrNull(result[0]!.providerModelId)).toBe(providerModelId)
  })

  it("does not treat a non-local catalog entry as a usable local offering", () => {
    const result = buildLocalInferenceSelections(
      models(model([{ configurationId, providerModelId }]), false),
      catalog(ProviderIdSchema.make("remote")),
      slots(),
    )
    expect(result).toEqual([])
  })
})
