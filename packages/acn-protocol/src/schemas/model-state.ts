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
const NonEmptyString = Schema.String.pipe(Schema.minLength(1))
const containsControlCharacter = (value: string) => /[\u0000-\u001f\u007f-\u009f]/u.test(value)

const isRealIsoCalendarDate = (value: string): boolean => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value)
  if (match === null) return false
  const year = Number(match[1])
  const month = Number(match[2])
  const day = Number(match[3])
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  return year >= 1 && month >= 1 && month <= 12 && day >= 1 && day <= daysInMonth[month - 1]!
}

export const ModelReleaseDateSchema = Schema.String.pipe(
  Schema.filter(isRealIsoCalendarDate, { message: () => "model release date must be a real YYYY-MM-DD calendar date" }),
  Schema.brand("ModelReleaseDate"),
)
export type ModelReleaseDate = typeof ModelReleaseDateSchema.Type

export const SlotIdSchema = Schema.Literal("primary", "secondary").pipe(Schema.brand("SlotId"))
export type SlotId = typeof SlotIdSchema.Type

export const PRIMARY_SLOT_ID = SlotIdSchema.make("primary")
export const SECONDARY_SLOT_ID = SlotIdSchema.make("secondary")
export const MODEL_SLOT_IDS = [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID] as const

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
  code: NonEmptyString,
  message: NonEmptyString,
  retryable: Schema.Boolean,
})
export type ModelFailure = typeof ModelFailureSchema.Type

export const LowMemoryModelInstanceFailureSchema = Schema.TaggedStruct("LowMemory", {
  code: Schema.Literal("low_memory"),
  message: Schema.String,
  retryable: Schema.Boolean,
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
  allocationHeadroomBytes: NonNegativeSafeInteger,
  systemReserveBytes: NonNegativeSafeInteger,
  loadBoundaryBytes: NonNegativeSafeInteger,
  minimumAdditionalAvailableBytes: PositiveSafeInteger,
  parallelSequences: PositiveSafeInteger,
})
export type LowMemoryModelInstanceFailure =
  typeof LowMemoryModelInstanceFailureSchema.Type

export const ModelInstanceFailureSchema = Schema.Union(
  ModelFailureSchema,
  LowMemoryModelInstanceFailureSchema,
)
export type ModelInstanceFailure = typeof ModelInstanceFailureSchema.Type

// =============================================================================
// Local model runtime facts and presentation
// =============================================================================

export const ModelVariantLabelSchema = NonEmptyString.pipe(Schema.brand("ModelVariantLabel"))
export type ModelVariantLabel = typeof ModelVariantLabelSchema.Type

export const formatModelDisplayName = (
  displayName: string,
  variantLabel: Option.Option<ModelVariantLabel>,
): string => Option.match(variantLabel, {
  onNone: () => displayName,
  onSome: (label) => `${displayName} (${label})`,
})

export const ModelLoadPlanSchema = Schema.Struct({
  contextWindowTokens: PositiveSafeInteger,
  parallelSequences: PositiveSafeInteger,
  physicalContextTokens: PositiveSafeInteger,
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
})
export type ModelLoadPlan = typeof ModelLoadPlanSchema.Type

export const ModelInstanceAllocationSchema = Schema.Struct({
  contextWindowTokens: PositiveSafeInteger,
  parallelSequences: PositiveSafeInteger,
  physicalContextTokens: PositiveSafeInteger,
  memoryDomains: Schema.Array(Schema.Struct({
    memoryDomainId: LocalInferenceMemoryDomainIdSchema,
    modelBytes: NonNegativeSafeInteger,
    contextBytes: NonNegativeSafeInteger,
    computeBytes: NonNegativeSafeInteger,
    auxiliaryBytes: NonNegativeSafeInteger,
  })),
})
export type ModelInstanceAllocation = typeof ModelInstanceAllocationSchema.Type

export const ModelStoppingAllocationSchema = Schema.Union(
  Schema.TaggedStruct("Planned", {
    allocation: Schema.optionalWith(ModelLoadPlanSchema, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Resident", {
    allocation: ModelInstanceAllocationSchema,
  }),
)
export type ModelStoppingAllocation = typeof ModelStoppingAllocationSchema.Type

export const CatalogBaseIdSchema = NonEmptyString.pipe(
  Schema.filter((value) => value !== "hf"
    && !value.includes(":")
    && !value.includes("/")
    && !value.includes("\\")
    && !containsControlCharacter(value)
    && value !== "."
    && value !== ".."
    && value.normalize("NFC") === value, {
    message: () => "catalog model ID must contain one identity component",
  }),
  Schema.brand("CatalogBaseId"),
)
export type CatalogBaseId = typeof CatalogBaseIdSchema.Type

export const CatalogVariantIdSchema = NonEmptyString.pipe(
  Schema.filter((value) => {
    const components = value.split(":")
    return components.length === 2 && components.every((component) => component.length > 0
      && component !== "."
      && component !== ".."
      && !component.includes("/")
      && !component.includes("\\")
      && !containsControlCharacter(component)
      && component.normalize("NFC") === component)
  }, { message: () => "catalog variant ID must be format:quality" }),
  Schema.brand("CatalogVariantId"),
)
export type CatalogVariantId = typeof CatalogVariantIdSchema.Type

export const HuggingFaceRepositoryIdSchema = NonEmptyString.pipe(
  Schema.filter((value) => {
    const components = value.split("/")
    return components.length === 2 && components.every((component) => component.length > 0
      && component !== "."
      && component !== ".."
      && !component.includes("\\")
      && !containsControlCharacter(component)
      && component.normalize("NFC") === component)
  }, { message: () => "Hugging Face repository ID must be owner/repository" }),
  Schema.brand("HuggingFaceRepositoryId"),
)
export type HuggingFaceRepositoryId = typeof HuggingFaceRepositoryIdSchema.Type

export const HuggingFaceArtifactSelectorSchema = NonEmptyString.pipe(
  Schema.filter((value) => !value.startsWith("/")
    && value.toLowerCase().endsWith(".gguf")
    && value.split("/").every((component) => component.length > 0
      && component !== "."
      && component !== ".."
      && !component.includes("\\")
      && !containsControlCharacter(component)
      && component.normalize("NFC") === component), {
    message: () => "Hugging Face artifact selector must be a normalized repository-relative GGUF path",
  }),
  Schema.brand("HuggingFaceArtifactSelector"),
)
export type HuggingFaceArtifactSelector = typeof HuggingFaceArtifactSelectorSchema.Type

const modelIdForm = (value: string): "Catalog" | "HuggingFace" | undefined => {
  if (value.startsWith("hf:")) {
    const components = value.slice(3).split("/")
    return components.length >= 3
      && Schema.is(HuggingFaceRepositoryIdSchema)(components.slice(0, 2).join("/"))
      && Schema.is(HuggingFaceArtifactSelectorSchema)(components.slice(2).join("/"))
      ? "HuggingFace"
      : undefined
  }
  const components = value.split(":")
  return components.length === 3
    && Schema.is(CatalogBaseIdSchema)(components[0])
    && Schema.is(CatalogVariantIdSchema)(components.slice(1).join(":"))
    ? "Catalog"
    : undefined
}

export const ModelIdSchema = ProviderModelIdSchema.pipe(
  Schema.filter((value) => modelIdForm(value) !== undefined, {
    message: () => "model ID must be a canonical catalog or Hugging Face model ID",
  }),
  Schema.brand("ModelId"),
)
export type ModelId = typeof ModelIdSchema.Type

export type ParsedModelId =
  | { readonly _tag: "Catalog"; readonly baseId: CatalogBaseId; readonly variantId: CatalogVariantId }
  | {
      readonly _tag: "HuggingFace"
      readonly repositoryId: HuggingFaceRepositoryId
      readonly artifactSelector: HuggingFaceArtifactSelector
    }

export const parseModelId = (modelId: ModelId): ParsedModelId => {
  if (modelId.startsWith("hf:")) {
    const components = modelId.slice(3).split("/")
    return {
      _tag: "HuggingFace",
      repositoryId: Schema.decodeUnknownSync(HuggingFaceRepositoryIdSchema)(components.slice(0, 2).join("/")),
      artifactSelector: Schema.decodeUnknownSync(HuggingFaceArtifactSelectorSchema)(components.slice(2).join("/")),
    }
  }
  const [baseId, format, quality] = modelId.split(":")
  return {
    _tag: "Catalog",
    baseId: Schema.decodeUnknownSync(CatalogBaseIdSchema)(baseId),
    variantId: Schema.decodeUnknownSync(CatalogVariantIdSchema)(`${format}:${quality}`),
  }
}

export const CatalogFormModelIdSchema = ModelIdSchema.pipe(Schema.filter(
  (modelId) => modelIdForm(modelId) === "Catalog",
  { message: () => "catalog model row requires a catalog-form model ID" },
), Schema.brand("CatalogFormModelId"))
export type CatalogFormModelId = typeof CatalogFormModelIdSchema.Type

export const HuggingFaceFormModelIdSchema = ModelIdSchema.pipe(Schema.filter(
  (modelId) => modelIdForm(modelId) === "HuggingFace",
  { message: () => "discovered model row requires a Hugging Face model ID" },
), Schema.brand("HuggingFaceFormModelId"))
export type HuggingFaceFormModelId = typeof HuggingFaceFormModelIdSchema.Type

export const ModelAssessmentIdSchema =
  NonEmptyString.pipe(Schema.brand("ModelAssessmentId"))
export type ModelAssessmentId = typeof ModelAssessmentIdSchema.Type

export const AssessmentEnvironmentIdSchema =
  NonEmptyString.pipe(Schema.brand("AssessmentEnvironmentId"))
export type AssessmentEnvironmentId = typeof AssessmentEnvironmentIdSchema.Type

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

export const ServingProfileSchema = Schema.Struct({
  contextLength: PositiveSafeInteger,
})
export type ServingProfile = typeof ServingProfileSchema.Type

export const ModelMetadataSchema = Schema.Struct({
  format: NonEmptyString,
  architecture: NonEmptyString,
  quantization: NonEmptyString,
  quantizationName: NonEmptyString,
  storageBytes: NonNegativeSafeInteger,
  maximumContextLength: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
})
export type ModelMetadata = typeof ModelMetadataSchema.Type

const DenseModelParameterizationSchema = Schema.Struct({
  architecture: Schema.Literal("dense"),
  totalParameters: PositiveSafeInteger,
})

const MixtureOfExpertsModelParameterizationSchema = Schema.Struct({
  architecture: Schema.Literal("mixtureOfExperts"),
  totalParameters: PositiveSafeInteger,
  activeParameters: PositiveSafeInteger,
}).pipe(Schema.filter(
  ({ activeParameters, totalParameters }) => activeParameters < totalParameters,
  { message: () => "active parameters must be less than total parameters" },
))

export const ModelParameterizationSchema = Schema.Union(
  DenseModelParameterizationSchema,
  MixtureOfExpertsModelParameterizationSchema,
)
export type ModelParameterization = typeof ModelParameterizationSchema.Type

const IntelligenceAsOfDateSchema = Schema.String.pipe(
  Schema.filter(isRealIsoCalendarDate, {
    message: () => "intelligence observation date must be a real YYYY-MM-DD calendar date",
  }),
  Schema.brand("IntelligenceAsOfDate"),
)

export const HttpsUrlSchema = Schema.String.pipe(
  Schema.filter((value) => {
    try {
      return new URL(value).protocol === "https:"
    } catch {
      return false
    }
  }, { message: () => "URL must be an absolute HTTPS URL" }),
  Schema.brand("HttpsUrl"),
)

export const IntelligenceProvenanceSchema = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal("artificialAnalysisIntelligenceIndex"),
    methodologyVersion: NonEmptyString,
    asOfDate: IntelligenceAsOfDateSchema,
    url: HttpsUrlSchema,
  }),
  Schema.Struct({
    kind: Schema.Literal("estimate"),
    target: Schema.Literal("artificialAnalysisIntelligenceIndex"),
    methodologyVersion: NonEmptyString,
    asOfDate: IntelligenceAsOfDateSchema,
    confidence: Schema.Literal("high", "moderate", "low"),
    methodology: NonEmptyString,
    evidenceUrls: Schema.NonEmptyArray(HttpsUrlSchema),
  }),
)
export type IntelligenceProvenance = typeof IntelligenceProvenanceSchema.Type

export const CatalogIntelligenceSchema = Schema.Struct({
  score: FiniteNonNegative,
  provenance: IntelligenceProvenanceSchema,
})
export type CatalogIntelligence = typeof CatalogIntelligenceSchema.Type

export const MemoryAssessmentSchema = Schema.Struct({
  memoryDomainId: LocalInferenceMemoryDomainIdSchema,
  capacityBytes: NonNegativeSafeInteger,
  requiredBytes: NonNegativeSafeInteger,
  compatibilityReserveBytes: NonNegativeSafeInteger,
  remainingBytes: Schema.Number.pipe(Schema.int()),
})
export type MemoryAssessment = typeof MemoryAssessmentSchema.Type

export const GenerationPerformanceEvidenceSchema = Schema.Struct({
  contextTokens: PositiveSafeInteger,
  lowerTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
  estimatedTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
  upperTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
  confidence: Schema.Literal("high", "moderate", "low"),
}).pipe(Schema.filter((sample) =>
  sample.lowerTokensPerSecond <= sample.estimatedTokensPerSecond
  && sample.estimatedTokensPerSecond <= sample.upperTokensPerSecond,
{ message: () => "performance rates must be ordered lower, expected, upper" }))
export type GenerationPerformanceEvidence =
  typeof GenerationPerformanceEvidenceSchema.Type

export const GenerationPerformanceSamplesSchema = Schema.Array(
  GenerationPerformanceEvidenceSchema,
).pipe(Schema.filter((samples) =>
  samples.length > 0
  && samples.every((sample, index) => index === 0
    || samples[index - 1]!.contextTokens < sample.contextTokens),
{ message: () => "performance samples must be nonempty and ordered by unique context" }))
export type GenerationPerformanceSamples = typeof GenerationPerformanceSamplesSchema.Type

export const localModelPerformanceContexts = (contextLength: number): readonly number[] => [...new Set([
  ...[25_000, 50_000, 75_000].filter((value) => value <= contextLength),
  contextLength,
])].sort((left, right) => left - right)

export const ResolvedModelInstallationSchema = Schema.TaggedStruct("Resolved", {
  installedBytes: NonNegativeSafeInteger,
  primaryPath: NonEmptyString,
  ownership: Schema.Literal("Magnitude", "ExternalHuggingFace", "Mixed"),
})
export type ResolvedModelInstallation = typeof ResolvedModelInstallationSchema.Type

export const UnresolvedModelInstallationSchema = Schema.TaggedStruct("Unresolved", {
  installedBytes: NonNegativeSafeInteger,
  ownership: Schema.Literal("Magnitude", "ExternalHuggingFace", "Mixed"),
})
export type UnresolvedModelInstallation = typeof UnresolvedModelInstallationSchema.Type

export const ModelInstallationSchema = Schema.Union(
  ResolvedModelInstallationSchema,
  UnresolvedModelInstallationSchema,
)
export type ModelInstallation = typeof ModelInstallationSchema.Type

export const ModelReleaseReasonSchema = Schema.Literal(
  "user_stop",
  "idle_timeout",
  "replacement",
  "memory_pressure",
)
export type ModelReleaseReason = typeof ModelReleaseReasonSchema.Type

export const ModelResidencySchema = Schema.Union(
  Schema.TaggedStruct("Unloaded", {}),
  Schema.TaggedStruct("Requested", {}),
  Schema.TaggedStruct("Loading", {
    stage: Schema.Literal("queued", "resolving", "unloading", "loading", "verifying"),
    progress: Schema.optionalWith(Schema.Number.pipe(Schema.finite(), Schema.between(0, 1)), {
      as: "Option",
      exact: true,
    }),
    plannedAllocation: Schema.optionalWith(ModelLoadPlanSchema, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Ready", {
    allocation: ModelInstanceAllocationSchema,
  }),
  Schema.TaggedStruct("Stopping", {
    reason: ModelReleaseReasonSchema,
    allocation: ModelStoppingAllocationSchema,
  }),
  Schema.TaggedStruct("Failed", { failure: ModelInstanceFailureSchema }),
)
export type ModelResidency = typeof ModelResidencySchema.Type

export const ModelTransferProgressSchema = Schema.Struct({
  stage: Schema.Literal("queued", "resolving", "checking_space", "downloading", "verifying", "publishing"),
  completedBytes: NonNegativeSafeInteger,
  totalBytes: NonNegativeSafeInteger,
  bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
}).pipe(Schema.filter((progress) => progress.completedBytes <= progress.totalBytes,
  { message: () => "model transfer progress cannot exceed its declared total" }))
export type ModelTransferProgress = typeof ModelTransferProgressSchema.Type

export const ModelAcquisitionFailureSchema = Schema.Union(
  Schema.TaggedStruct("Interrupted", {}),
  Schema.TaggedStruct("InsufficientDiskSpace", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("SourceUnavailable", {}),
  Schema.TaggedStruct("NetworkUnavailable", {}),
  Schema.TaggedStruct("LocalStorageFailure", {}),
  Schema.TaggedStruct("CorruptDownload", {}),
  Schema.TaggedStruct("Internal", { message: NonEmptyString }),
)
export type ModelAcquisitionFailure = typeof ModelAcquisitionFailureSchema.Type

const InstalledModelFields = {
  installation: ModelInstallationSchema,
  residencyState: ModelResidencySchema,
} as const

/**
 * The single per-model materialization lifecycle: what exists on disk, the one
 * transfer that may be running for it, and — once bits exist — the model's
 * runtime residency. Every variant is a reachable product state; progress and
 * failure payloads exist only under the states they belong to.
 */
export const LocalModelAcquisitionStateSchema = Schema.Union(
  Schema.TaggedStruct("NotInstalled", {}),
  Schema.TaggedStruct("Installing", { progress: ModelTransferProgressSchema }),
  /** Acknowledged failure returns to NotInstalled. */
  Schema.TaggedStruct("InstallFailed", { failure: ModelAcquisitionFailureSchema }),
  Schema.TaggedStruct("Installed", InstalledModelFields),
  Schema.TaggedStruct("UpdateAvailable", InstalledModelFields),
  Schema.TaggedStruct("Updating", {
    ...InstalledModelFields,
    progress: ModelTransferProgressSchema,
  }),
  /** Acknowledged failure returns to UpdateAvailable. */
  Schema.TaggedStruct("UpdateFailed", {
    ...InstalledModelFields,
    failure: ModelAcquisitionFailureSchema,
  }),
  Schema.TaggedStruct("Removing", InstalledModelFields),
  Schema.TaggedStruct("RemoveFailed", {
    ...InstalledModelFields,
    failure: ModelFailureSchema,
  }),
)
export type LocalModelAcquisitionState = typeof LocalModelAcquisitionStateSchema.Type

export type InstalledLocalModelAcquisitionState = Extract<
  LocalModelAcquisitionState,
  { readonly installation: unknown }
>

/** The installed-family payload when any version of the model is on disk. */
export const installedAcquisition = (
  state: LocalModelAcquisitionState,
): InstalledLocalModelAcquisitionState | undefined => state._tag === "Installed"
  || state._tag === "UpdateAvailable"
  || state._tag === "Updating"
  || state._tag === "UpdateFailed"
  || state._tag === "Removing"
  || state._tag === "RemoveFailed"
  ? state
  : undefined

/** The transfer progress when a download is running for the model. */
export const acquisitionProgress = (
  state: LocalModelAcquisitionState,
): ModelTransferProgress | undefined => state._tag === "Installing" || state._tag === "Updating"
  ? state.progress
  : undefined

/** The unacknowledged transfer failure when the model's last download failed. */
export const acquisitionFailure = (
  state: LocalModelAcquisitionState,
): ModelAcquisitionFailure | undefined => state._tag === "InstallFailed" || state._tag === "UpdateFailed"
  ? state.failure
  : undefined

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
  variantLabel: Schema.optionalWith(ModelVariantLabelSchema, { as: "Option", exact: true }),
  supportedSlots: Schema.Array(SlotIdSchema),
  contextWindow: PositiveSafeInteger,
  maxOutputTokens: PositiveSafeInteger,
  memory: Schema.optionalWith(Schema.Array(MemoryAssessmentSchema), { as: "Option", exact: true }),
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

export const LocalModelCatalogDataSchema = Schema.Struct({
  releaseDate: ModelReleaseDateSchema,
  parameterization: ModelParameterizationSchema,
  intelligence: CatalogIntelligenceSchema,
  fidelityRank: NonNegativeSafeInteger,
  quantizationAware: Schema.Boolean,
})
export type LocalModelCatalogData = typeof LocalModelCatalogDataSchema.Type

export const DiscoveredModelCatalogAttributionSchema = Schema.Union(
  Schema.TaggedStruct("NotInCatalog", {}),
  Schema.TaggedStruct("AttributionFailed", { failure: ModelFailureSchema }),
)
export type DiscoveredModelCatalogAttribution =
  typeof DiscoveredModelCatalogAttributionSchema.Type

const NormalizedRankingScoreSchema = Schema.Number.pipe(
  Schema.finite(),
  Schema.between(0, 1),
)

export const LocalModelRankingScoresSchema = Schema.Struct({
  intelligence: NormalizedRankingScoreSchema,
  speed: NormalizedRankingScoreSchema,
  fidelity: NormalizedRankingScoreSchema,
})
export type LocalModelRankingScores = typeof LocalModelRankingScoresSchema.Type

export const LocalModelMemoryHeadroomObservationSchema = Schema.Struct({
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
  allocationHeadroomBytes: NonNegativeSafeInteger,
  abortReserveBytes: NonNegativeSafeInteger,
  loadBoundaryBytes: NonNegativeSafeInteger,
})
export type LocalModelMemoryHeadroomObservation =
  typeof LocalModelMemoryHeadroomObservationSchema.Type

export const LocalModelSystemMemoryUseStateSchema = Schema.Union(
  Schema.TaggedStruct("NotObserved", {}),
  Schema.TaggedStruct("WithinRecommendedHeadroom", {
    recommendedHeadroomBytes: NonNegativeSafeInteger,
    predictedHeadroomBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("High", {
    recommendedHeadroomBytes: NonNegativeSafeInteger,
    predictedHeadroomBytes: NonNegativeSafeInteger,
  }),
)
export type LocalModelSystemMemoryUseState = typeof LocalModelSystemMemoryUseStateSchema.Type

export const LocalModelCurrentHeadroomStateSchema = Schema.Union(
  Schema.TaggedStruct("NotObserved", {}),
  Schema.TaggedStruct("Sufficient", {
    observation: LocalModelMemoryHeadroomObservationSchema,
  }),
  Schema.TaggedStruct("Insufficient", {
    observation: LocalModelMemoryHeadroomObservationSchema,
    minimumAdditionalAvailableBytes: PositiveSafeInteger,
  }),
)
export type LocalModelCurrentHeadroomState =
  typeof LocalModelCurrentHeadroomStateSchema.Type

export const LocalModelMemorySchema = Schema.Struct({
  domains: Schema.Array(MemoryAssessmentSchema),
  totalRequiredBytes: NonNegativeSafeInteger,
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
  systemUseState: LocalModelSystemMemoryUseStateSchema,
  currentHeadroomState: LocalModelCurrentHeadroomStateSchema,
}).pipe(Schema.filter((memory) => {
  const uniqueDomains = new Set(memory.domains.map(({ memoryDomainId }) => memoryDomainId)).size
    === memory.domains.length
  const totalRequiredBytes = memory.domains.reduce((total, domain) => total + domain.requiredBytes, 0)
  const requiredSystemMemoryBytes = memory.domains
    .filter(({ memoryDomainId }) => memoryDomainId === "system")
    .reduce((total, domain) => total + domain.requiredBytes, 0)
  return uniqueDomains
    && memory.totalRequiredBytes === totalRequiredBytes
    && memory.requiredSystemMemoryBytes === requiredSystemMemoryBytes
}, { message: () => "local model memory totals and domains must agree with their evidence" }))
export type LocalModelMemory = typeof LocalModelMemorySchema.Type

export const LocalModelFitsAssessmentSchema = Schema.TaggedStruct("Fits", {
  assessmentId: ModelAssessmentIdSchema,
  environmentId: AssessmentEnvironmentIdSchema,
  profile: ServingProfileSchema,
  memory: LocalModelMemorySchema,
  performance: GenerationPerformanceSamplesSchema,
})
export type LocalModelFitsAssessment = typeof LocalModelFitsAssessmentSchema.Type

export const LocalModelDoesNotFitAssessmentSchema = Schema.TaggedStruct("DoesNotFit", {
  assessmentId: ModelAssessmentIdSchema,
  environmentId: AssessmentEnvironmentIdSchema,
  profile: ServingProfileSchema,
  memoryDomains: Schema.Array(MemoryAssessmentSchema),
  totalRequiredBytes: NonNegativeSafeInteger,
  deficitBytes: NonNegativeSafeInteger,
  limitingResource: NonEmptyString,
}).pipe(Schema.filter((assessment) => assessment.totalRequiredBytes === assessment.memoryDomains
  .reduce((total, domain) => total + domain.requiredBytes, 0),
{ message: () => "non-fitting assessment memory total must match its domain evidence" }))
export type LocalModelDoesNotFitAssessment = typeof LocalModelDoesNotFitAssessmentSchema.Type

export const LocalModelIncompatibleAssessmentSchema = Schema.TaggedStruct("Incompatible", {
  environmentId: AssessmentEnvironmentIdSchema,
  profile: ServingProfileSchema,
  failure: ModelFailureSchema,
})
export type LocalModelIncompatibleAssessment = typeof LocalModelIncompatibleAssessmentSchema.Type

export const LocalModelAssessmentSchema = Schema.Union(
  LocalModelFitsAssessmentSchema,
  LocalModelDoesNotFitAssessmentSchema,
  LocalModelIncompatibleAssessmentSchema,
)
export type LocalModelAssessment = typeof LocalModelAssessmentSchema.Type

export const LocalModelPresentationSchema = Schema.Struct({
  displayName: NonEmptyString,
  variantLabel: ModelVariantLabelSchema,
  description: Schema.String,
  license: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  sourceUrls: Schema.Array(HttpsUrlSchema),
}).pipe(Schema.filter((presentation) => new Set(presentation.sourceUrls).size === presentation.sourceUrls.length,
  { message: () => "local model source URLs must be unique" }))
export type LocalModelPresentation = typeof LocalModelPresentationSchema.Type

export const SpeculativeMethodSchema = Schema.Union(
  Schema.TaggedStruct("Mtp", {}),
  Schema.TaggedStruct("DFlash", {}),
  Schema.TaggedStruct("DSpark", {}),
)
export type SpeculativeMethod = typeof SpeculativeMethodSchema.Type

const LocalModelAssessingServingStateSchema = Schema.TaggedStruct("Assessing", {
  profile: ServingProfileSchema,
})
const LocalModelFailedServingStateSchema = Schema.TaggedStruct("Failed", {
  profile: Schema.optionalWith(ServingProfileSchema, {
    as: "Option",
    exact: true,
  }),
  failure: ModelFailureSchema,
})
const DiscoveredModelFailedServingStateSchema = Schema.TaggedStruct("Failed", {
  profile: ServingProfileSchema,
  failure: ModelFailureSchema,
})
const LocalModelAssessedFields = {
  metadata: ModelMetadataSchema,
  capabilities: ModelCapabilitiesSchema,
  speculativeMethod: Schema.optionalWith(SpeculativeMethodSchema, { as: "Option", exact: true }),
} as const
const CatalogModelAssessedFitsServingStateSchema = Schema.TaggedStruct("Assessed", {
  ...LocalModelAssessedFields,
  assessment: LocalModelFitsAssessmentSchema,
  rankingScores: Schema.optionalWith(LocalModelRankingScoresSchema, {
    as: "Option",
    exact: true,
  }),
})
const CatalogModelAssessedUnavailableServingStateSchema = Schema.TaggedStruct("Assessed", {
  ...LocalModelAssessedFields,
  assessment: Schema.Union(
    LocalModelDoesNotFitAssessmentSchema,
    LocalModelIncompatibleAssessmentSchema,
  ),
})
const DiscoveredModelAssessedServingStateSchema = Schema.TaggedStruct("Assessed", {
  ...LocalModelAssessedFields,
  assessment: LocalModelAssessmentSchema,
})
export const CatalogLocalModelServingStateSchema = Schema.Union(
  LocalModelAssessingServingStateSchema,
  LocalModelFailedServingStateSchema,
  CatalogModelAssessedFitsServingStateSchema,
  CatalogModelAssessedUnavailableServingStateSchema,
)
export type CatalogLocalModelServingState = typeof CatalogLocalModelServingStateSchema.Type

export const DiscoveredLocalModelServingStateSchema = Schema.Union(
  LocalModelAssessingServingStateSchema,
  DiscoveredModelFailedServingStateSchema,
  DiscoveredModelAssessedServingStateSchema,
)
export type DiscoveredLocalModelServingState = typeof DiscoveredLocalModelServingStateSchema.Type

export const LocalModelServingStateSchema = Schema.Union(
  CatalogLocalModelServingStateSchema,
  DiscoveredLocalModelServingStateSchema,
)
export type LocalModelServingState = typeof LocalModelServingStateSchema.Type

const LocalModelFields = {
  presentation: LocalModelPresentationSchema,
} as const

const DiscoveredReadyLocalModelStateSchema = Schema.TaggedStruct("Ready", {
  installation: ResolvedModelInstallationSchema,
  residencyState: ModelResidencySchema,
  catalogAttribution: DiscoveredModelCatalogAttributionSchema,
  servingState: DiscoveredLocalModelServingStateSchema,
})
const DiscoveredUnavailableLocalModelStateSchema = Schema.TaggedStruct("Unavailable", {
  installation: ResolvedModelInstallationSchema,
  failure: ModelFailureSchema,
})
export const DiscoveredLocalModelStateSchema = Schema.Union(
  DiscoveredReadyLocalModelStateSchema,
  DiscoveredUnavailableLocalModelStateSchema,
)
export type DiscoveredLocalModelState = typeof DiscoveredLocalModelStateSchema.Type

export const CatalogLocalModelSchema = Schema.TaggedStruct("Catalog", {
  ...LocalModelFields,
  modelId: CatalogFormModelIdSchema,
  servingState: CatalogLocalModelServingStateSchema,
  catalogData: LocalModelCatalogDataSchema,
  storageBytes: NonNegativeSafeInteger,
  acquisitionState: LocalModelAcquisitionStateSchema,
})
export type CatalogLocalModel = typeof CatalogLocalModelSchema.Type

export const DiscoveredLocalModelSchema = Schema.TaggedStruct("Discovered", {
  ...LocalModelFields,
  modelId: HuggingFaceFormModelIdSchema,
  state: DiscoveredLocalModelStateSchema,
})
export type DiscoveredLocalModel = typeof DiscoveredLocalModelSchema.Type

export const LocalModelSchema = Schema.Union(CatalogLocalModelSchema, DiscoveredLocalModelSchema)
export type LocalModel = typeof LocalModelSchema.Type

export const LocalModelPreparationSchema = Schema.Struct({
  discovery: Schema.Struct({
    complete: Schema.Boolean,
    modelsFound: NonNegativeSafeInteger,
  }),
  assessment: Schema.Struct({
    complete: Schema.Boolean,
    settledModels: NonNegativeSafeInteger,
    totalModels: NonNegativeSafeInteger,
  }).pipe(Schema.filter(({ settledModels, totalModels }) => settledModels <= totalModels, {
    message: () => "settled model assessments cannot exceed total model assessments",
  })),
}).pipe(Schema.filter(({ assessment }) => !assessment.complete
  || assessment.settledModels === assessment.totalModels, {
  message: () => "complete model assessment progress must have every target settled",
}))
export type LocalModelPreparation = typeof LocalModelPreparationSchema.Type

export const LocalModelsStateSchema = Schema.Struct({
  preparation: LocalModelPreparationSchema,
  models: Schema.Array(LocalModelSchema),
}).pipe(Schema.filter(({ models }) => {
  const identities = models.map(({ modelId }) => modelId)
  return new Set(identities).size === identities.length
}, { message: () => "local models must have unique canonical model IDs" }))
export type LocalModelsState = typeof LocalModelsStateSchema.Type

export const LocalProviderOfferingSchema = Schema.Struct({
  providerModelId: ModelIdSchema,
  profile: ServingProfileSchema,
  capabilities: ModelCapabilitiesSchema,
})
export type LocalProviderOffering = typeof LocalProviderOfferingSchema.Type

export const ProviderAuthenticationSchema = Schema.Literal("Authenticated", "NotConfigured", "NotRequired")
export const ProviderKindSchema = Schema.Literal("Hosted", "Local", "Custom")
export type ProviderKind = typeof ProviderKindSchema.Type
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
  kind: ProviderKindSchema,
  authentication: ProviderAuthenticationSchema,
  availability: ProviderAvailabilitySchema,
})
export type ProviderCatalogEntry = typeof ProviderCatalogEntrySchema.Type

export const ProviderModelIdentitySchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
})
export type ProviderModelIdentity = typeof ProviderModelIdentitySchema.Type

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

export const ModelCatalogEntrySchema = Schema.Union(
  Schema.TaggedStruct("Remote", {
    offering: ProviderModelCatalogEntrySchema,
  }),
  Schema.TaggedStruct("Local", {
    product: LocalModelSchema,
    offering: Schema.optionalWith(ProviderModelCatalogEntrySchema, {
      as: "Option",
      exact: true,
    }),
  }),
)
export type ModelCatalogEntry = typeof ModelCatalogEntrySchema.Type

const ModelCatalogSnapshotFields = {
  providers: Schema.Array(ProviderCatalogEntrySchema),
  models: Schema.Array(ModelCatalogEntrySchema),
  failures: Schema.Array(ProviderCatalogFailureSchema),
  localModelPreparation: LocalModelPreparationSchema,
} as const

export const ModelCatalogStateSchema = Schema.Union(
  Schema.TaggedStruct("Initializing", {}),
  Schema.TaggedStruct("Ready", ModelCatalogSnapshotFields),
  Schema.TaggedStruct("Refreshing", ModelCatalogSnapshotFields),
  Schema.TaggedStruct("Degraded", ModelCatalogSnapshotFields),
).pipe(Schema.filter((state) => state._tag === "Initializing" || (() => {
  const identities = state.models.map((entry) => entry._tag === "Remote"
    ? `${entry.offering.providerId}:${entry.offering.providerModelId}`
    : `local:${entry.product.modelId}`)
  return new Set(identities).size === identities.length
})(), { message: () => "model catalog entries must have unique provider-qualified identities" }))
export type ModelCatalogState = typeof ModelCatalogStateSchema.Type

export const SlotSelectionSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  reasoningEffort: ReasoningEffortSchema,
})
export type SlotSelection = typeof SlotSelectionSchema.Type

/** Durable Magnitude model intent. Runtime availability and residency are ICN projections. */
export const ModelSlotSelectionsStateSchema = Schema.Struct({
  slots: Schema.Struct({
    primary: Schema.optionalWith(SlotSelectionSchema, { as: "Option", exact: true }),
    secondary: Schema.optionalWith(SlotSelectionSchema, { as: "Option", exact: true }),
  }),
  recentModels: Schema.Struct({
    primary: Schema.Array(ProviderModelIdentitySchema),
    secondary: Schema.Array(ProviderModelIdentitySchema),
  }),
  favoriteModels: Schema.Array(ProviderModelIdentitySchema),
})
export type ModelSlotSelectionsState = typeof ModelSlotSelectionsStateSchema.Type

export class ModelSlotUnassigned extends Schema.TaggedClass<ModelSlotUnassigned>()("Unassigned", {
  slotId: SlotIdSchema,
}) {}

export class ModelSlotResolving extends Schema.TaggedClass<ModelSlotResolving>()("Resolving", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
}) {}

export const ModelSlotDescriptorSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  displayName: NonEmptyString,
  variantLabel: Schema.optionalWith(ModelVariantLabelSchema, { as: "Option", exact: true }),
})
export type ModelSlotDescriptor = typeof ModelSlotDescriptorSchema.Type

export const ModelSlotAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("Pending", {}),
  Schema.TaggedStruct("Available", {}),
  Schema.TaggedStruct("Unavailable", { failure: ModelFailureSchema }),
)
export type ModelSlotAvailability = typeof ModelSlotAvailabilitySchema.Type

export const ModelSlotActionSchema = Schema.Literal("Load", "Stop", "RetryLoad")
export type ModelSlotAction = typeof ModelSlotActionSchema.Type

export const modelSlotActions = (
  availability: ModelSlotAvailability,
  residency: ModelResidency,
): readonly ModelSlotAction[] => {
  switch (residency._tag) {
    case "Requested":
    case "Loading":
    case "Ready":
      return ["Stop"]
    case "Stopping":
      return []
    case "Failed":
      return residency.failure.retryable ? ["RetryLoad"] : []
    case "Unloaded":
      return availability._tag === "Available" ? ["Load"] : []
  }
}

export class ModelSlotConfiguredRemote extends Schema.TaggedClass<ModelSlotConfiguredRemote>()("ConfiguredRemote", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
  descriptor: ModelSlotDescriptorSchema,
  availability: ModelSlotAvailabilitySchema,
  actions: Schema.Array(ModelSlotActionSchema),
}) {}

export class ModelSlotConfiguredLocal extends Schema.TaggedClass<ModelSlotConfiguredLocal>()("ConfiguredLocal", {
  slotId: SlotIdSchema,
  selection: SlotSelectionSchema,
  descriptor: ModelSlotDescriptorSchema,
  availability: ModelSlotAvailabilitySchema,
  residency: ModelResidencySchema,
  actions: Schema.Array(ModelSlotActionSchema),
}) {}

export const ModelSlotLifecycle = defineFSM(
  {
    Unassigned: ModelSlotUnassigned,
    Resolving: ModelSlotResolving,
    ConfiguredRemote: ModelSlotConfiguredRemote,
    ConfiguredLocal: ModelSlotConfiguredLocal,
  },
  {
    Unassigned: ["Resolving", "ConfiguredRemote", "ConfiguredLocal"],
    Resolving: ["Unassigned", "ConfiguredRemote", "ConfiguredLocal"],
    ConfiguredRemote: ["Unassigned", "Resolving", "ConfiguredLocal"],
    ConfiguredLocal: ["Unassigned", "Resolving", "ConfiguredRemote"],
  } as const,
)

export const ModelSlotSchema = Schema.Union(
  ModelSlotUnassigned,
  ModelSlotResolving,
  ModelSlotConfiguredRemote,
  ModelSlotConfiguredLocal,
).pipe(Schema.filter((slot) => {
  if (slot._tag === "Unassigned" || slot._tag === "Resolving") return true
  if (slot.selection.providerId !== slot.descriptor.providerId
    || slot.selection.providerModelId !== slot.descriptor.providerModelId) return false
  if (slot._tag === "ConfiguredRemote") {
    return slot.selection.providerId !== "local" && slot.actions.length === 0
  }
  if (slot.selection.providerId !== "local") return false
  const expectedActions = modelSlotActions(slot.availability, slot.residency)
  return expectedActions.length === slot.actions.length
    && expectedActions.every((action, index) => action === slot.actions[index])
}, { message: () => "slot identity, provider kind, and actions must agree with canonical slot state" }))
export type ModelSlot = typeof ModelSlotSchema.Type

export const ModelSlotsStateSchema = Schema.Struct({
  slots: Schema.Struct({
    primary: ModelSlotSchema,
    secondary: ModelSlotSchema,
  }),
  recentModels: Schema.Struct({
    primary: Schema.Array(ProviderModelIdentitySchema),
    secondary: Schema.Array(ProviderModelIdentitySchema),
  }),
  favoriteModels: Schema.Array(ProviderModelIdentitySchema),
}).pipe(
  Schema.filter((state) => state.slots.primary.slotId === PRIMARY_SLOT_ID
    && state.slots.secondary.slotId === SECONDARY_SLOT_ID,
  { message: () => "each model slot state must carry its containing slot identity" }),
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
  productName: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  processor: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  logicalCores: PositiveSafeInteger,
  totalSystemMemoryBytes: NonNegativeSafeInteger,
  availableSystemMemoryBytes: NonNegativeSafeInteger,
  systemAllocationCapacityBytes: NonNegativeSafeInteger,
  systemAllocationHeadroomBytes: NonNegativeSafeInteger,
  abortReserveBytes: NonNegativeSafeInteger,
  accelerators: Schema.Array(LocalInferenceAcceleratorSchema),
  memoryDomains: Schema.Array(LocalInferenceMemoryDomainSchema),
}).pipe(Schema.filter((hardware) => {
  const memoryDomainIds = hardware.memoryDomains.map(({ memoryDomainId }) => memoryDomainId)
  const acceleratorIds = hardware.accelerators.map(({ acceleratorId }) => acceleratorId)
  const domains = new Set(memoryDomainIds)
  return domains.size === memoryDomainIds.length
    && new Set(acceleratorIds).size === acceleratorIds.length
    && hardware.accelerators.every(({ memoryDomainId }) => domains.has(memoryDomainId))
}, { message: () => "hardware identities must be unique and accelerator memory-domain references must resolve" }))
export type LocalInferenceHardware = typeof LocalInferenceHardwareSchema.Type
