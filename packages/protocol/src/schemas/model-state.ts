import { Option, Schema } from "effect"
import {
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/ai/provider/model"
import { FSM } from "@magnitudedev/utils"

const { defineFSM } = FSM

const NonNegativeSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.nonNegative(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)
const PositiveSafeInteger = NonNegativeSafeInteger.pipe(Schema.positive())
const FiniteNonNegative = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())

export const SlotIdSchema = Schema.Literal("primary", "secondary").pipe(Schema.brand("SlotId"))
export type SlotId = typeof SlotIdSchema.Type

export const PRIMARY_SLOT_ID = SlotIdSchema.make("primary")
export const SECONDARY_SLOT_ID = SlotIdSchema.make("secondary")
export const MODEL_SLOT_IDS = [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID] as const

export const LocalModelIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(512),
  Schema.brand("LocalModelId"),
)
export type LocalModelId = typeof LocalModelIdSchema.Type

export const LocalInferenceAcceleratorIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(512),
  Schema.brand("LocalInferenceAcceleratorId"),
)
export type LocalInferenceAcceleratorId = typeof LocalInferenceAcceleratorIdSchema.Type

export const LocalInferenceMemoryDomainIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(512),
  Schema.brand("LocalInferenceMemoryDomainId"),
)
export type LocalInferenceMemoryDomainId = typeof LocalInferenceMemoryDomainIdSchema.Type

export const PercentageSchema = Schema.Number.pipe(Schema.int(), Schema.between(0, 100))
export type Percentage = typeof PercentageSchema.Type

export const ModelFailureSchema = Schema.Struct({
  code: Schema.String,
  message: Schema.String,
  retryable: Schema.Boolean,
})
export type ModelFailure = typeof ModelFailureSchema.Type

const ModelReasoningCapabilitiesSchema = Schema.Struct({
  supported: Schema.Boolean,
  efforts: Schema.Array(ReasoningEffortSchema),
  defaultEffort: Schema.optionalWith(ReasoningEffortSchema, { as: "Option", exact: true }),
}).pipe(Schema.filter((reasoning) => {
  const unique = new Set(reasoning.efforts).size === reasoning.efforts.length
  if (!unique) return false
  if (!reasoning.supported) return reasoning.efforts.length === 0 && reasoning.defaultEffort._tag === "None"
  return reasoning.efforts.length > 0
    && reasoning.defaultEffort._tag === "Some"
    && reasoning.efforts.includes(reasoning.defaultEffort.value)
}, { message: () => "reasoning capabilities must have a unique, internally consistent effort set" }))

export const ModelCapabilitiesSchema = Schema.Struct({
  vision: Schema.Boolean,
  tools: Schema.Boolean,
  structuredOutput: Schema.Boolean,
  reasoning: ModelReasoningCapabilitiesSchema,
})
export type ModelCapabilities = typeof ModelCapabilitiesSchema.Type

export const LocalModelFitSchema = Schema.Union(
  Schema.TaggedStruct("Fits", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
    memoryDomainIds: Schema.Array(LocalInferenceMemoryDomainIdSchema),
  }),
  Schema.TaggedStruct("DoesNotFit", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
    limitingResource: Schema.String,
    memoryDomainIds: Schema.Array(LocalInferenceMemoryDomainIdSchema),
  }),
)
export type LocalModelFit = typeof LocalModelFitSchema.Type

export const ProviderAuthenticationSchema = Schema.Literal("Authenticated", "NotConfigured", "NotRequired")
export const ProviderAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("Available", {}),
  Schema.TaggedStruct("Loading", {
    message: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("NotFound", {
    message: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    hint: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
)
export const ProviderCatalogEntrySchema = Schema.Struct({
  providerId: ProviderIdSchema,
  displayName: Schema.String,
  authentication: ProviderAuthenticationSchema,
  availability: ProviderAvailabilitySchema,
})
export type ProviderCatalogEntry = typeof ProviderCatalogEntrySchema.Type

export const ProviderModelDisabledReasonSchema = Schema.Literal(
  "insufficient_resources",
  "provider_unavailable",
  "model_unavailable",
  "installation_unavailable",
  "incompatible_runtime",
  "invalid_configuration",
)
export type ProviderModelDisabledReason = typeof ProviderModelDisabledReasonSchema.Type

export const ProviderModelCatalogEntrySchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  modelFamilyId: Schema.optionalWith(ModelFamilyIdSchema, { as: "Option", exact: true }),
  displayName: Schema.String,
  supportedSlots: Schema.Array(SlotIdSchema),
  contextWindow: PositiveSafeInteger,
  maxOutputTokens: PositiveSafeInteger,
  capabilities: ModelCapabilitiesSchema,
  availability: Schema.Union(
    Schema.TaggedStruct("Available", {}),
    Schema.TaggedStruct("Disabled", { reason: ProviderModelDisabledReasonSchema }),
  ),
  pricing: Schema.optionalWith(Schema.Struct({
    input: FiniteNonNegative,
    output: FiniteNonNegative,
    cachedInput: Schema.optionalWith(FiniteNonNegative, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
}).pipe(Schema.filter((model) => new Set(model.supportedSlots).size === model.supportedSlots.length,
  { message: () => "supported model slots must be unique" }))
export type ProviderModelCatalogEntry = typeof ProviderModelCatalogEntrySchema.Type

export const ProviderCatalogFailureSchema = Schema.Union(
  Schema.TaggedStruct("ProviderFailure", {
    providerId: ProviderIdSchema,
    message: Schema.String,
  }),
  Schema.TaggedStruct("CatalogFailure", {
    message: Schema.String,
  }),
)
export type ProviderCatalogFailure = typeof ProviderCatalogFailureSchema.Type

const ProviderCatalogSnapshotFields = {
  providers: Schema.Array(ProviderCatalogEntrySchema),
  models: Schema.Array(ProviderModelCatalogEntrySchema),
} as const

export class ProviderModelCatalogLoading extends Schema.TaggedClass<ProviderModelCatalogLoading>()("Loading", {}) {}
export class ProviderModelCatalogReady extends Schema.TaggedClass<ProviderModelCatalogReady>()("Ready", ProviderCatalogSnapshotFields) {}
export class ProviderModelCatalogRefreshing extends Schema.TaggedClass<ProviderModelCatalogRefreshing>()("Refreshing", {
  ...ProviderCatalogSnapshotFields,
  failures: Schema.Array(ProviderCatalogFailureSchema),
}) {}
export class ProviderModelCatalogDegraded extends Schema.TaggedClass<ProviderModelCatalogDegraded>()("Degraded", {
  ...ProviderCatalogSnapshotFields,
  failures: Schema.Array(ProviderCatalogFailureSchema),
}) {}
export class ProviderModelCatalogUnavailable extends Schema.TaggedClass<ProviderModelCatalogUnavailable>()("Unavailable", {
  providers: Schema.Array(ProviderCatalogEntrySchema),
  failures: Schema.Array(ProviderCatalogFailureSchema),
}) {}

export const ProviderModelCatalogLifecycle = defineFSM(
  {
    Loading: ProviderModelCatalogLoading,
    Ready: ProviderModelCatalogReady,
    Refreshing: ProviderModelCatalogRefreshing,
    Degraded: ProviderModelCatalogDegraded,
    Unavailable: ProviderModelCatalogUnavailable,
  },
  {
    Loading: ["Ready", "Degraded", "Unavailable"],
    Ready: ["Refreshing"],
    Refreshing: ["Ready", "Degraded", "Unavailable"],
    Degraded: ["Refreshing"],
    Unavailable: ["Refreshing"],
  } as const,
)

export const ProviderModelCatalogStateSchema = Schema.Union(
  ProviderModelCatalogLoading,
  ProviderModelCatalogReady,
  ProviderModelCatalogRefreshing,
  ProviderModelCatalogDegraded,
  ProviderModelCatalogUnavailable,
).pipe(Schema.filter((state) => {
  if (state._tag === "Loading") return true
  const providerIds = state.providers.map(({ providerId }) => providerId)
  const uniqueProviderIds = new Set(providerIds)
  if (uniqueProviderIds.size !== providerIds.length) return false
  if (state._tag === "Unavailable") return true
  const modelIdsByProvider = new Map<typeof ProviderIdSchema.Type, Set<typeof ProviderModelIdSchema.Type>>()
  for (const { providerId, providerModelId } of state.models) {
    const modelIds = modelIdsByProvider.get(providerId) ?? new Set<typeof ProviderModelIdSchema.Type>()
    if (modelIds.has(providerModelId)) return false
    modelIds.add(providerModelId)
    modelIdsByProvider.set(providerId, modelIds)
  }
  return state.models.every(({ providerId }) => uniqueProviderIds.has(providerId))
}, { message: () => "catalog identities must be unique and every model provider must resolve" }))
export type ProviderModelCatalogState = typeof ProviderModelCatalogStateSchema.Type

export const LocalModelInventoryEntryDetailsSchema = Schema.Struct({
  localModelId: LocalModelIdSchema,
  providerModelId: ProviderModelIdSchema,
  modelFamilyId: Schema.optionalWith(ModelFamilyIdSchema, { as: "Option", exact: true }),
  displayName: Schema.String,
  family: Schema.String,
  architecture: Schema.Literal("Dense", "MixtureOfExperts"),
  capabilities: ModelCapabilitiesSchema,
  contextWindow: PositiveSafeInteger,
  maxOutputTokens: PositiveSafeInteger,
  quantization: Schema.String,
  downloadBytes: NonNegativeSafeInteger,
  fit: LocalModelFitSchema,
  recommendation: Schema.optionalWith(Schema.Struct({
    intent: Schema.Literal("balanced", "best_quality", "fastest", "lightweight"),
    explanation: Schema.String,
    fidelityLabel: Schema.String,
    fidelityEvidence: Schema.String,
    repository: Schema.String,
    revision: Schema.String,
    files: Schema.Array(Schema.Struct({ path: Schema.String, sha256: Schema.String })),
    sourcePageUrl: Schema.String,
    estimatedRuntimeBytes: NonNegativeSafeInteger,
    fitMarginBytes: Schema.Number.pipe(Schema.finite()),
    estimatedGeneration: Schema.optionalWith(Schema.Struct({
      contextTokens: PositiveSafeInteger,
      lowerTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
      expectedTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
      upperTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
      confidence: Schema.Literal("high", "moderate", "low"),
      method: Schema.String,
    }), { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
})
export type LocalModelInventoryEntryDetails = typeof LocalModelInventoryEntryDetailsSchema.Type

export class LocalModelAvailableForDownload extends Schema.TaggedClass<LocalModelAvailableForDownload>()("AvailableForDownload", {
  model: LocalModelInventoryEntryDetailsSchema,
}) {}
export class LocalModelDownloading extends Schema.TaggedClass<LocalModelDownloading>()("Downloading", {
  model: LocalModelInventoryEntryDetailsSchema,
  percentage: PercentageSchema,
  completedBytes: NonNegativeSafeInteger,
  totalBytes: NonNegativeSafeInteger,
}) {}
export class LocalModelDownloaded extends Schema.TaggedClass<LocalModelDownloaded>()("Downloaded", {
  model: LocalModelInventoryEntryDetailsSchema,
  downloadedBytes: NonNegativeSafeInteger,
}) {}
export class LocalModelDownloadFailed extends Schema.TaggedClass<LocalModelDownloadFailed>()("DownloadFailed", {
  model: LocalModelInventoryEntryDetailsSchema,
  completedBytes: NonNegativeSafeInteger,
  totalBytes: NonNegativeSafeInteger,
  error: ModelFailureSchema,
}) {}

export const LocalModelInventoryEntryLifecycle = defineFSM(
  {
    AvailableForDownload: LocalModelAvailableForDownload,
    Downloading: LocalModelDownloading,
    Downloaded: LocalModelDownloaded,
    DownloadFailed: LocalModelDownloadFailed,
  },
  {
    AvailableForDownload: ["Downloading"],
    Downloading: ["Downloaded", "DownloadFailed"],
    Downloaded: ["AvailableForDownload"],
    DownloadFailed: ["Downloading"],
  } as const,
)

export const LocalModelInventoryEntrySchema = Schema.Union(
  LocalModelAvailableForDownload,
  LocalModelDownloading,
  LocalModelDownloaded,
  LocalModelDownloadFailed,
).pipe(Schema.filter((entry) => entry._tag !== "Downloading" && entry._tag !== "DownloadFailed"
  || entry.completedBytes <= entry.totalBytes,
{ message: () => "completed download bytes cannot exceed total bytes" }))
export type LocalModelInventoryEntry = typeof LocalModelInventoryEntrySchema.Type

export class LocalModelInventoryLoading extends Schema.TaggedClass<LocalModelInventoryLoading>()("Loading", {}) {}
export class LocalModelInventoryReady extends Schema.TaggedClass<LocalModelInventoryReady>()("Ready", {
  entries: Schema.Array(LocalModelInventoryEntrySchema),
}) {}
export class LocalModelInventoryFailed extends Schema.TaggedClass<LocalModelInventoryFailed>()("Failed", {
  error: ModelFailureSchema,
}) {}

export const LocalModelInventoryLifecycle = defineFSM(
  {
    Loading: LocalModelInventoryLoading,
    Ready: LocalModelInventoryReady,
    Failed: LocalModelInventoryFailed,
  },
  {
    Loading: ["Ready", "Failed"],
    Ready: ["Failed"],
    Failed: ["Loading"],
  } as const,
)

export const LocalModelInventoryStateSchema = Schema.Union(
  LocalModelInventoryLoading,
  LocalModelInventoryReady,
  LocalModelInventoryFailed,
).pipe(Schema.filter((state) => state._tag !== "Ready"
  || new Set(state.entries.map((entry) => entry.model.localModelId)).size === state.entries.length,
{ message: () => "local model inventory identities must be unique" }))
export type LocalModelInventoryState = typeof LocalModelInventoryStateSchema.Type

export const SlotSelectionSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  reasoningEffort: ReasoningEffortSchema,
})
export type SlotSelection = typeof SlotSelectionSchema.Type

export const ModelSlotBlockedReasonSchema = Schema.Union(
  Schema.TaggedStruct("ProviderUnavailable", { message: Schema.String }),
  Schema.TaggedStruct("ModelUnavailable", { message: Schema.String }),
  Schema.TaggedStruct("InvalidConfiguration", { message: Schema.String }),
  Schema.TaggedStruct("LocalModelLoadFailed", { error: ModelFailureSchema }),
)
export type ModelSlotBlockedReason = typeof ModelSlotBlockedReasonSchema.Type

export class ModelSlotUnassigned extends Schema.TaggedClass<ModelSlotUnassigned>()("Unassigned", {
  slotId: SlotIdSchema,
}) {}
export class ModelSlotUnloadedLocalModel extends Schema.TaggedClass<ModelSlotUnloadedLocalModel>()("UnloadedLocalModel", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
}) {}
export class ModelSlotLoadingLocalModel extends Schema.TaggedClass<ModelSlotLoadingLocalModel>()("LoadingLocalModel", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
  percentage: PercentageSchema,
}) {}
export class ModelSlotReady extends Schema.TaggedClass<ModelSlotReady>()("Ready", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
}) {}
export class ModelSlotUnloadingLocalModel extends Schema.TaggedClass<ModelSlotUnloadingLocalModel>()("UnloadingLocalModel", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
}) {}
export class ModelSlotBlocked extends Schema.TaggedClass<ModelSlotBlocked>()("Blocked", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
  reason: ModelSlotBlockedReasonSchema,
}) {}

export const ModelSlotLifecycle = defineFSM(
  {
    Unassigned: ModelSlotUnassigned,
    UnloadedLocalModel: ModelSlotUnloadedLocalModel,
    LoadingLocalModel: ModelSlotLoadingLocalModel,
    Ready: ModelSlotReady,
    UnloadingLocalModel: ModelSlotUnloadingLocalModel,
    Blocked: ModelSlotBlocked,
  },
  {
    Unassigned: ["UnloadedLocalModel", "LoadingLocalModel", "Ready", "UnloadingLocalModel", "Blocked"],
    UnloadedLocalModel: ["Unassigned", "LoadingLocalModel", "Ready", "Blocked"],
    LoadingLocalModel: ["Unassigned", "Ready", "UnloadedLocalModel", "UnloadingLocalModel", "Blocked"],
    Ready: ["Unassigned", "UnloadingLocalModel", "UnloadedLocalModel", "Blocked"],
    UnloadingLocalModel: ["Unassigned", "UnloadedLocalModel", "LoadingLocalModel", "Ready", "Blocked"],
    Blocked: ["Unassigned", "UnloadedLocalModel", "LoadingLocalModel", "Ready"],
  } as const,
)

export const ModelSlotSchema = Schema.Union(
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotUnloadingLocalModel,
  ModelSlotBlocked,
).pipe(Schema.filter((slot) => slot._tag === "Unassigned"
  || slot._tag === "Ready"
  || slot._tag === "Blocked"
  || slot.selection.providerId === "local",
{ message: () => "local-model slot states require the local provider" }))
export type ModelSlot = typeof ModelSlotSchema.Type

export const ModelSlotsStateSchema = Schema.Struct({
  slots: Schema.Struct({
    primary: ModelSlotSchema,
    secondary: ModelSlotSchema,
  }),
}).pipe(
  Schema.filter((state) => state.slots.primary.slotId === PRIMARY_SLOT_ID
    && state.slots.secondary.slotId === SECONDARY_SLOT_ID,
  { message: () => "each model slot state must carry its containing slot identity" }),
  Schema.filter((state) => {
    const activeLocalModels = [state.slots.primary, state.slots.secondary].flatMap((slot) =>
      (slot._tag === "LoadingLocalModel" || slot._tag === "Ready" || slot._tag === "UnloadingLocalModel")
        && slot.selection.providerId === "local"
        ? [slot.selection.providerModelId]
        : [])
    return new Set(activeLocalModels).size <= 1
  }, { message: () => "at most one distinct local model may be active" }),
)
export type ModelSlotsState = typeof ModelSlotsStateSchema.Type

export const LocalInferenceAcceleratorSchema = Schema.Struct({
  acceleratorId: LocalInferenceAcceleratorIdSchema,
  name: Schema.String,
  backend: Schema.String,
  memoryDomainId: LocalInferenceMemoryDomainIdSchema,
})
export const LocalInferenceMemoryDomainSchema = Schema.Struct({
  memoryDomainId: LocalInferenceMemoryDomainIdSchema,
  kind: Schema.Literal("System", "PhysicalDevice", "UnifiedMemory"),
  totalBytes: NonNegativeSafeInteger,
  stableCapacityBytes: NonNegativeSafeInteger,
  availableBytes: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  sharesSystemMemory: Schema.Boolean,
})
export const LocalInferenceHardwareSchema = Schema.Struct({
  platform: Schema.Literal("MacOS", "Linux", "Windows"),
  architecture: Schema.Literal("Arm64", "X64"),
  processor: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  logicalCores: PositiveSafeInteger,
  totalSystemMemoryBytes: NonNegativeSafeInteger,
  availableSystemMemoryBytes: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  accelerators: Schema.Array(LocalInferenceAcceleratorSchema),
  memoryDomains: Schema.Array(LocalInferenceMemoryDomainSchema),
  residentMemory: Schema.optionalWith(Schema.Struct({
    domains: Schema.Array(Schema.Struct({
      memoryDomainId: LocalInferenceMemoryDomainIdSchema,
      modelBytes: NonNegativeSafeInteger,
      contextBytes: NonNegativeSafeInteger,
      computeBytes: NonNegativeSafeInteger,
      auxiliaryBytes: NonNegativeSafeInteger,
    })),
  }), { as: "Option", exact: true }),
}).pipe(Schema.filter((hardware) => {
  const memoryDomainIds = hardware.memoryDomains.map(({ memoryDomainId }) => memoryDomainId)
  const acceleratorIds = hardware.accelerators.map(({ acceleratorId }) => acceleratorId)
  const domains = new Set(memoryDomainIds)
  return domains.size === memoryDomainIds.length
    && new Set(acceleratorIds).size === acceleratorIds.length
    && hardware.accelerators.every(({ memoryDomainId }) => domains.has(memoryDomainId))
    && Option.match(hardware.residentMemory, {
      onNone: () => true,
      onSome: ({ domains: residentDomains }) =>
        residentDomains.every(({ memoryDomainId }) => domains.has(memoryDomainId)),
    })
}, { message: () => "hardware identities must be unique and accelerator memory-domain references must resolve" }))
export type LocalInferenceHardware = typeof LocalInferenceHardwareSchema.Type
