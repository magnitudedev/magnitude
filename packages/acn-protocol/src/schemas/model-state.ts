import { Option, Schema } from "effect"
import {
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/ai/provider/model"
import { FSM } from "@magnitudedev/utils"
import { HuggingFaceAccessSchema } from "./hugging-face"

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

/** Stable identity of one inference execution device, including the host CPU. */
export const LocalInferenceDeviceIdSchema = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(512),
  Schema.brand("LocalInferenceDeviceId"),
)
export type LocalInferenceDeviceId = typeof LocalInferenceDeviceIdSchema.Type

export const LocalInferenceBackendSchema = Schema.Literal("cpu", "metal", "cuda", "vulkan")
export type LocalInferenceBackend = typeof LocalInferenceBackendSchema.Type

export const PercentageSchema = Schema.Number.pipe(Schema.int(), Schema.between(0, 100))
export type Percentage = typeof PercentageSchema.Type

export const ModelFailureSchema = Schema.Struct({
  code: NonEmptyString,
  message: NonEmptyString,
  retryable: Schema.Boolean,
})
export type ModelFailure = typeof ModelFailureSchema.Type

/** Why a model that fits the machine could not be given its memory. */
export const MemoryShortageSchema = Schema.Union(
  /** Memory that cannot be moved is in the way. `availableBytes` is what a load may claim. */
  Schema.TaggedStruct("Blocked", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
  }),
  /** The machine was displacing other programs' memory; no byte count describes it. */
  Schema.TaggedStruct("UnderPressure", {}),
)
export type MemoryShortage = typeof MemoryShortageSchema.Type

export const MemoryShortageModelInstanceFailureSchema = Schema.TaggedStruct("MemoryShortage", {
  code: Schema.Literal("memory_shortage"),
  message: Schema.String,
  retryable: Schema.Boolean,
  shortage: MemoryShortageSchema,
})
export type MemoryShortageModelInstanceFailure =
  typeof MemoryShortageModelInstanceFailureSchema.Type

export const ModelInstanceFailureSchema = Schema.Union(
  ModelFailureSchema,
  MemoryShortageModelInstanceFailureSchema,
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

export const ModelLoadDeviceSchema = Schema.Struct({
  deviceId: LocalInferenceDeviceIdSchema,
  backend: LocalInferenceBackendSchema,
})
export type ModelLoadDevice = typeof ModelLoadDeviceSchema.Type

export const ModelLoadPlanSchema = Schema.Struct({
  contextWindowTokens: PositiveSafeInteger,
  /** Startup peak claim in the selected device's allocation domain; excludes context growth. */
  requiredMemoryBytes: NonNegativeSafeInteger,
  device: ModelLoadDeviceSchema,
})
export type ModelLoadPlan = typeof ModelLoadPlanSchema.Type

export const ModelInstanceAllocationSchema = Schema.Struct({
  contextWindowTokens: PositiveSafeInteger,
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

export const ModelHeaderDigestSchema = Schema.String.pipe(
  Schema.pattern(/^[0-9a-f]{64}$/, { message: () => "model header digest must be 64 lowercase hex characters" }),
  Schema.brand("ModelHeaderDigest"),
)
export type ModelHeaderDigest = typeof ModelHeaderDigestSchema.Type

type ModelIdForm = "Catalog" | "HuggingFace" | "Local"

const modelIdForm = (value: string): ModelIdForm | undefined => {
  if (value.startsWith("hf:")) {
    const components = value.slice(3).split("/")
    return components.length >= 3
      && Schema.is(HuggingFaceRepositoryIdSchema)(components.slice(0, 2).join("/"))
      && Schema.is(HuggingFaceArtifactSelectorSchema)(components.slice(2).join("/"))
      ? "HuggingFace"
      : undefined
  }
  if (value.startsWith("local:")) {
    return Schema.is(ModelHeaderDigestSchema)(value.slice(6)) ? "Local" : undefined
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
    message: () => "model ID must be a canonical catalog, Hugging Face, or local model ID",
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
  | { readonly _tag: "Local"; readonly headerDigest: ModelHeaderDigest }

export const parseModelId = (modelId: ModelId): ParsedModelId => {
  if (modelId.startsWith("hf:")) {
    const components = modelId.slice(3).split("/")
    return {
      _tag: "HuggingFace",
      repositoryId: Schema.decodeUnknownSync(HuggingFaceRepositoryIdSchema)(components.slice(0, 2).join("/")),
      artifactSelector: Schema.decodeUnknownSync(HuggingFaceArtifactSelectorSchema)(components.slice(2).join("/")),
    }
  }
  if (modelId.startsWith("local:")) {
    return { _tag: "Local", headerDigest: Schema.decodeUnknownSync(ModelHeaderDigestSchema)(modelId.slice(6)) }
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
  { message: () => "model ID must be a Hugging Face model ID" },
), Schema.brand("HuggingFaceFormModelId"))
export type HuggingFaceFormModelId = typeof HuggingFaceFormModelIdSchema.Type

export const LocalFormModelIdSchema = ModelIdSchema.pipe(Schema.filter(
  (modelId) => modelIdForm(modelId) === "Local",
  { message: () => "model ID must be a local header-digest model ID" },
), Schema.brand("LocalFormModelId"))
export type LocalFormModelId = typeof LocalFormModelIdSchema.Type

/** Any non-catalog model: `hf:` when it has a Hugging Face identity, `local:` otherwise. */
export const CustomFormModelIdSchema = ModelIdSchema.pipe(Schema.filter(
  (modelId) => modelIdForm(modelId) !== "Catalog",
  { message: () => "custom model row requires a Hugging Face or local model ID" },
), Schema.brand("CustomFormModelId"))
export type CustomFormModelId = typeof CustomFormModelIdSchema.Type

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

/** A model's intelligence as a whole percentage of the top Artificial Analysis Intelligence Index score. */
export const IntelligenceScoreSchema = Schema.Int.pipe(Schema.between(0, 100))

export const MemoryAssessmentSchema = Schema.Struct({
  memoryDomainId: LocalInferenceMemoryDomainIdSchema,
  capacityBytes: NonNegativeSafeInteger,
  requiredBytes: NonNegativeSafeInteger,
  compatibilityReserveBytes: NonNegativeSafeInteger,
  remainingBytes: Schema.Number.pipe(Schema.int()),
})
export type MemoryAssessment = typeof MemoryAssessmentSchema.Type

/** An estimated decode speed at one context depth: bytes per decode step over the device's bandwidth. */
export const GenerationPerformanceEvidenceSchema = Schema.Struct({
  contextTokens: PositiveSafeInteger,
  estimatedTokensPerSecond: Schema.Number.pipe(Schema.finite(), Schema.positive()),
})
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
  ownership: Schema.Literal("Magnitude", "External", "Mixed"),
})
export type ResolvedModelInstallation = typeof ResolvedModelInstallationSchema.Type

export const UnresolvedModelInstallationSchema = Schema.TaggedStruct("Unresolved", {
  installedBytes: NonNegativeSafeInteger,
  ownership: Schema.Literal("Magnitude", "External", "Mixed"),
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
  "failure",
)
export type ModelReleaseReason = typeof ModelReleaseReasonSchema.Type

export const ModelLoadStageSchema = Schema.Literal(
  "queued",
  "preparing",
  "optimizing",
  "loading_weights",
  "finalizing",
)
export type ModelLoadStage = typeof ModelLoadStageSchema.Type

export const ModelResidencySchema = Schema.Union(
  Schema.TaggedStruct("Unloaded", {}),
  /** Was loaded and is no longer, for `reason`. `Unloaded` is a model with no instance at all. */
  Schema.TaggedStruct("Stopped", { reason: ModelReleaseReasonSchema }),
  Schema.TaggedStruct("Requested", {}),
  Schema.TaggedStruct("Loading", {
    stage: ModelLoadStageSchema,
    /** Completed fraction by measured work: tuning (when the load tunes) fills the first half. */
    fraction: Schema.Number.pipe(Schema.finite(), Schema.between(0, 1)),
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

/** Kernel tuning after a download, in tuning work units; `preparing` counts them (0 of 0). */
export const ModelOptimizationProgressSchema = Schema.Struct({
  stage: Schema.Literal("preparing", "tuning"),
  completed: NonNegativeSafeInteger,
  total: NonNegativeSafeInteger,
  device: Schema.optionalWith(ModelLoadDeviceSchema, { as: "Option", exact: true }),
}).pipe(Schema.filter((progress) => progress.completed <= progress.total,
  { message: () => "model optimization progress cannot exceed its declared total" }))
export type ModelOptimizationProgress = typeof ModelOptimizationProgressSchema.Type

export const ModelAcquisitionFailureSchema = Schema.Union(
  Schema.TaggedStruct("Interrupted", {}),
  Schema.TaggedStruct("InsufficientDiskSpace", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("SourceUnavailable", {}),
  Schema.TaggedStruct("HuggingFaceAccessRequired", { access: HuggingFaceAccessSchema }),
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
  /** One-time kernel tuning after an install or update download; it ends Installed however it ends. */
  Schema.TaggedStruct("Optimizing", {
    ...InstalledModelFields,
    progress: ModelOptimizationProgressSchema,
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
  || state._tag === "Optimizing"
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
  "unsupported_model",
  "invalid_configuration",
  "catalog_disabled",
  "deprecated",
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

const CatalogDeprecationDateSchema = Schema.String.pipe(
  Schema.filter(isRealIsoCalendarDate, {
    message: () => "catalog deprecation date must be a real YYYY-MM-DD calendar date",
  }),
  Schema.brand("CatalogDeprecationDate"),
)

/** What the release promises for a catalog model, independent of this device's assessment. */
export const CatalogSupportSchema = Schema.Union(
  Schema.TaggedStruct("Supported", {}),
  Schema.TaggedStruct("Disabled", { reason: NonEmptyString }),
  Schema.TaggedStruct("Deprecated", {
    since: CatalogDeprecationDateSchema,
    replacement: CatalogFormModelIdSchema,
    reason: NonEmptyString,
  }),
)
export type CatalogSupport = typeof CatalogSupportSchema.Type

/**
 * A catalog model's catalog facts. `aliases` are the IDs its files would have as found models (the
 * catalog's exact target found in another app's location); a request naming one means this model. A
 * hidden external copy keeps its own ID and is not an alias here.
 */
export const LocalModelCatalogDataSchema = Schema.Struct({
  releaseDate: ModelReleaseDateSchema,
  parameterization: ModelParameterizationSchema,
  intelligence: IntelligenceScoreSchema,
  support: CatalogSupportSchema,
  fidelityRank: NonNegativeSafeInteger,
  quantizationAware: Schema.Boolean,
  aliases: Schema.Array(CustomFormModelIdSchema),
})
export type LocalModelCatalogData = typeof LocalModelCatalogDataSchema.Type

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

export const LocalModelMemorySchema = Schema.Struct({
  domains: Schema.Array(MemoryAssessmentSchema),
  totalRequiredBytes: NonNegativeSafeInteger,
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
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

/** A fitting model, with one decode-speed estimate per assessed context depth. */
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

export const WeightTypeCountSchema = Schema.Struct({
  ggmlType: NonEmptyString,
  tensorCount: PositiveSafeInteger,
})
export type WeightTypeCount = typeof WeightTypeCountSchema.Type

/** A reason case this protocol does not know yet; `kind` is the engine's tag. */
const UnknownUnsupportedReasonSchema = Schema.TaggedStruct("Unknown", { kind: NonEmptyString, detail: Schema.String })

const KnownUnsupportedReasonSchema = Schema.Union(
  Schema.TaggedStruct("Architecture", { architecture: NonEmptyString }),
  Schema.TaggedStruct("WeightTypes", { types: Schema.NonEmptyArray(WeightTypeCountSchema) }),
  Schema.TaggedStruct("RopeScaling", { kind: NonEmptyString }),
  Schema.TaggedStruct("UnrecognizedTensor", { name: NonEmptyString }),
  Schema.TaggedStruct("UnrecognizedMetadata", { key: NonEmptyString }),
  Schema.TaggedStruct("MissingMetadata", { key: NonEmptyString }),
  Schema.TaggedStruct("Tokenizer", {
    model: NonEmptyString,
    pre: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("ChatTemplate", {}),
  Schema.TaggedStruct("TiedOutputHead", {}),
  Schema.TaggedStruct("TargetMismatch", { detail: Schema.String }),
  Schema.TaggedStruct("KernelDomain", {
    backend: NonEmptyString,
    entry: NonEmptyString,
    statics: Schema.Array(Schema.Tuple(NonEmptyString, NonNegativeSafeInteger)),
  }),
  Schema.TaggedStruct("Backend", { backend: NonEmptyString }),
  Schema.TaggedStruct("Unreadable", {}),
  Schema.TaggedStruct("UnsupportedFormat", {}),
  UnknownUnsupportedReasonSchema,
)

const knownUnsupportedReasonTags: ReadonlySet<string> = new Set<UnsupportedReason["_tag"]>([
  "Architecture", "WeightTypes", "RopeScaling", "UnrecognizedTensor", "UnrecognizedMetadata", "MissingMetadata", "Tokenizer",
  "ChatTemplate", "TiedOutputHead", "TargetMismatch", "KernelDomain", "Backend", "Unreadable", "UnsupportedFormat",
  "Unknown",
])

const UnrecognizedUnsupportedReasonSchema = Schema.transform(
  Schema.Struct({
    _tag: NonEmptyString.pipe(Schema.filter((tag) => !knownUnsupportedReasonTags.has(tag))),
    detail: Schema.optional(Schema.String),
  }),
  Schema.typeSchema(UnknownUnsupportedReasonSchema),
  {
    strict: true,
    decode: ({ _tag, detail }) => ({ _tag: "Unknown" as const, kind: _tag, detail: detail ?? "" }),
    encode: ({ kind, detail }) => ({ _tag: kind, detail }),
  },
)

/**
 * Why the engine cannot run a model, one entry per distinct cause. A case added by a newer engine
 * decodes as `Unknown` so it still renders.
 */
export const UnsupportedReasonSchema = Schema.Union(
  KnownUnsupportedReasonSchema,
  UnrecognizedUnsupportedReasonSchema,
)
export type UnsupportedReason = typeof UnsupportedReasonSchema.Type

/**
 * The engine cannot execute a custom model or candidate; catalog models are never unsupported. An
 * unreadable file has no serving profile.
 */
export const LocalModelUnsupportedAssessmentSchema = Schema.TaggedStruct("Unsupported", {
  environmentId: AssessmentEnvironmentIdSchema,
  profile: Schema.optionalWith(ServingProfileSchema, { as: "Option", exact: true }),
  reasons: Schema.NonEmptyArray(UnsupportedReasonSchema),
  detail: Schema.String,
})
export type LocalModelUnsupportedAssessment = typeof LocalModelUnsupportedAssessmentSchema.Type

export const LocalModelAssessmentSchema = Schema.Union(
  LocalModelFitsAssessmentSchema,
  LocalModelDoesNotFitAssessmentSchema,
  LocalModelUnsupportedAssessmentSchema,
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
  Schema.TaggedStruct("DFlash2", {}),
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
const CustomModelFailedServingStateSchema = Schema.TaggedStruct("Failed", {
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
const CatalogModelAssessedDoesNotFitServingStateSchema = Schema.TaggedStruct("Assessed", {
  ...LocalModelAssessedFields,
  assessment: LocalModelDoesNotFitAssessmentSchema,
})
const CustomModelAssessedServingStateSchema = Schema.TaggedStruct("Assessed", {
  ...LocalModelAssessedFields,
  assessment: LocalModelAssessmentSchema,
})
export const CatalogLocalModelServingStateSchema = Schema.Union(
  LocalModelAssessingServingStateSchema,
  LocalModelFailedServingStateSchema,
  CatalogModelAssessedFitsServingStateSchema,
  CatalogModelAssessedDoesNotFitServingStateSchema,
)
export type CatalogLocalModelServingState = typeof CatalogLocalModelServingStateSchema.Type

/** Custom models are never ranked; `Unsupported` stays only on added models, whose files remain removable. */
export const CustomLocalModelServingStateSchema = Schema.Union(
  LocalModelAssessingServingStateSchema,
  CustomModelFailedServingStateSchema,
  CustomModelAssessedServingStateSchema,
)
export type CustomLocalModelServingState = typeof CustomLocalModelServingStateSchema.Type

export const LocalModelServingStateSchema = Schema.Union(
  CatalogLocalModelServingStateSchema,
  CustomLocalModelServingStateSchema,
)
export type LocalModelServingState = typeof LocalModelServingStateSchema.Type

// =============================================================================
// Custom models: origin, drafters, and candidates
// =============================================================================

export const FoundModelSourceSchema = Schema.Union(
  Schema.TaggedStruct("HuggingFaceCache", {}),
  Schema.TaggedStruct("LmStudio", {}),
  Schema.TaggedStruct("LlamaCpp", {}),
  Schema.TaggedStruct("Ollama", {}),
  Schema.TaggedStruct("Folder", { path: NonEmptyString }),
)
export type FoundModelSource = typeof FoundModelSourceSchema.Type

/** Where a custom model's files came from, and therefore who owns them. */
export const CustomModelOriginSchema = Schema.Union(
  Schema.TaggedStruct("Added", { repository: HuggingFaceRepositoryIdSchema, commit: NonEmptyString }),
  Schema.TaggedStruct("Imported", { path: NonEmptyString }),
  Schema.TaggedStruct("Found", { source: FoundModelSourceSchema }),
)
export type CustomModelOrigin = typeof CustomModelOriginSchema.Type

export const ModelLocationKindSchema = Schema.Literal("HuggingFaceCache", "LmStudio", "LlamaCpp", "Ollama", "Folder")
export type ModelLocationKind = typeof ModelLocationKindSchema.Type

export const AutomaticModelLocationKindSchema = Schema.Literal("HuggingFaceCache", "LmStudio", "LlamaCpp", "Ollama")
export type AutomaticModelLocationKind = typeof AutomaticModelLocationKindSchema.Type

/** Whether a location's directory exists and Magnitude has permission to read it. */
export const ModelLocationStateSchema = Schema.Literal("Found", "NotFound", "Unreadable")
export type ModelLocationState = typeof ModelLocationStateSchema.Type

/** One directory of a model location, and whether it exists and Magnitude can read it. */
export const ModelLocationPathSchema = Schema.Struct({
  path: NonEmptyString,
  state: ModelLocationStateSchema,
})
export type ModelLocationPath = typeof ModelLocationPathSchema.Type

/** A location's state across its directories: found if any is found, else unreadable if any is unreadable. */
export const modelLocationState = (paths: readonly ModelLocationPath[]): ModelLocationState =>
  paths.some(({ state }) => state === "Found") ? "Found"
    : paths.some(({ state }) => state === "Unreadable") ? "Unreadable"
    : "NotFound"

/**
 * Where Magnitude scans for GGUFs: an automatic location with every directory Magnitude resolved
 * for it on this host, most authoritative first, or a user folder, which is one directory.
 */
export const ModelLocationSchema = Schema.Struct({
  kind: ModelLocationKindSchema,
  paths: Schema.NonEmptyArray(ModelLocationPathSchema),
  enabled: Schema.Boolean,
  automatic: Schema.Boolean,
  state: ModelLocationStateSchema,
}).pipe(
  Schema.filter((location) => location.automatic === (location.kind !== "Folder"),
    { message: () => "only user folders are not automatic locations" }),
  Schema.filter((location) => location.automatic || location.paths.length === 1,
    { message: () => "a user folder is one directory" }),
  Schema.filter((location) => location.state === modelLocationState(location.paths),
    { message: () => "a location's state summarizes its directories' states" }),
)
export type ModelLocation = typeof ModelLocationSchema.Type

/** The engine's answer for a drafter paired with a target. */
export const DrafterVerdictSchema = Schema.Union(
  Schema.TaggedStruct("None", {}),
  Schema.TaggedStruct("Bound", { method: SpeculativeMethodSchema, memoryBytes: NonNegativeSafeInteger }),
  Schema.TaggedStruct("Rejected", {
    reasons: Schema.NonEmptyArray(UnsupportedReasonSchema),
    detail: Schema.String,
  }),
)
export type DrafterVerdict = typeof DrafterVerdictSchema.Type

/** Which drafter: a catalog model's own drafter, MTP heads inside the target, or a GGUF elsewhere. */
export const DrafterReferenceSchema = Schema.Union(
  Schema.TaggedStruct("Catalog", { modelId: CatalogFormModelIdSchema }),
  Schema.TaggedStruct("Embedded", {}),
  Schema.TaggedStruct("HuggingFace", {
    repository: HuggingFaceRepositoryIdSchema,
    commit: NonEmptyString,
    path: HuggingFaceArtifactSelectorSchema,
  }),
  Schema.TaggedStruct("Local", { path: NonEmptyString }),
)
export type DrafterReference = typeof DrafterReferenceSchema.Type

/**
 * Where a drafter candidate comes from, in automatic-choice order: a separate drafter in the model's
 * own repository or folder, the matched catalog model's drafter, MTP heads built into the model, then
 * a drafter the user entered, which is never chosen automatically.
 */
export const DrafterCandidateCategorySchema = Schema.Literal(
  "NextToModel",
  "CatalogDrafter",
  "BuiltIn",
  "Other",
)
export type DrafterCandidateCategory = typeof DrafterCandidateCategorySchema.Type

export const DrafterLineageSchema = Schema.Literal("Exact", "Lineage", "Fingerprint", "None")
export type DrafterLineage = typeof DrafterLineageSchema.Type

export const DrafterCandidateSchema = Schema.Struct({
  category: DrafterCandidateCategorySchema,
  reference: DrafterReferenceSchema,
  name: NonEmptyString,
  label: NonEmptyString,
  sizeBytes: NonNegativeSafeInteger,
  lineage: DrafterLineageSchema,
  verdict: DrafterVerdictSchema,
  automatic: Schema.Boolean,
  onThisComputer: Schema.Boolean,
  method: Schema.optionalWith(SpeculativeMethodSchema, { as: "Option", exact: true }),
})
export type DrafterCandidate = typeof DrafterCandidateSchema.Type

/**
 * What a sync started: the first download of a model not installed (or the resumption of one that
 * failed or was interrupted), an update of an installed model or its drafter to a newer revision,
 * a missing drafter of an installed model, or nothing because the model is current.
 */
export const LocalModelSyncOutcomeSchema = Schema.Literal("Installing", "Updating", "DownloadingDrafter", "AlreadyCurrent")
export type LocalModelSyncOutcome = typeof LocalModelSyncOutcomeSchema.Type

/** A requested drafter: the preselected one, none, or a specific reference. */
export const DrafterChoiceSchema = Schema.Union(
  Schema.TaggedStruct("Auto", {}),
  Schema.TaggedStruct("None", {}),
  Schema.TaggedStruct("Reference", { reference: DrafterReferenceSchema }),
)
export type DrafterChoice = typeof DrafterChoiceSchema.Type

/** Whether an attached drafter's files are here; a drafter-only download reports its own progress. */
export const AttachedDrafterAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("Present", {}),
  Schema.TaggedStruct("Downloading", { progress: ModelTransferProgressSchema }),
  Schema.TaggedStruct("DownloadFailed", { failure: ModelAcquisitionFailureSchema }),
)
export type AttachedDrafterAvailability = typeof AttachedDrafterAvailabilitySchema.Type

/** The drafter attached to a custom model; `updateAvailable` means a newer catalog drafter revision exists. */
export const AttachedDrafterSchema = Schema.Struct({
  reference: DrafterReferenceSchema,
  name: NonEmptyString,
  label: NonEmptyString,
  sizeBytes: NonNegativeSafeInteger,
  verdict: DrafterVerdictSchema,
  availability: AttachedDrafterAvailabilitySchema,
  updateAvailable: Schema.Boolean,
})
export type AttachedDrafter = typeof AttachedDrafterSchema.Type

export const CustomModelPresentationSchema = Schema.Struct({
  displayName: NonEmptyString,
  variantLabel: ModelVariantLabelSchema,
  sourceLabel: NonEmptyString,
  variantOf: Schema.optionalWith(CatalogFormModelIdSchema, { as: "Option", exact: true }),
  license: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  sourceUrls: Schema.Array(HttpsUrlSchema),
}).pipe(Schema.filter((presentation) => new Set(presentation.sourceUrls).size === presentation.sourceUrls.length,
  { message: () => "custom model source URLs must be unique" }))
export type CustomModelPresentation = typeof CustomModelPresentationSchema.Type

const ExternalModelAcquisitionStateSchema = Schema.TaggedStruct("External", {
  installation: ResolvedModelInstallationSchema,
  residencyState: ModelResidencySchema,
})
const UnavailableModelAcquisitionStateSchema = Schema.TaggedStruct("Unavailable", {
  installation: ResolvedModelInstallationSchema,
  failure: ModelFailureSchema,
})

/**
 * An added model's managed lifecycle, or for imported and found models, files Magnitude references
 * but never owns: `External` when readable, `Unavailable` when not.
 */
export const CustomModelAcquisitionStateSchema = Schema.Union(
  ...LocalModelAcquisitionStateSchema.members,
  ExternalModelAcquisitionStateSchema,
  UnavailableModelAcquisitionStateSchema,
)
export type CustomModelAcquisitionState = typeof CustomModelAcquisitionStateSchema.Type

const isManagedAcquisitionState = (state: CustomModelAcquisitionState): state is LocalModelAcquisitionState =>
  state._tag !== "External" && state._tag !== "Unavailable"

const LocalModelFields = {
  presentation: LocalModelPresentationSchema,
} as const

/** A part of a catalog model fetched on its own: the projector that adds image input, or the catalog drafter. */
export const CatalogComponentSchema = Schema.Literal("Projector", "Drafter")
export type CatalogComponent = typeof CatalogComponentSchema.Type

/**
 * A component a catalog model lacks because its exact target was found on this computer without it. The model
 * serves without it, with every catalog component that is here, until it is fetched; it is never an update.
 */
export const CatalogMissingComponentSchema = Schema.Struct({
  component: CatalogComponentSchema,
  downloadBytes: NonNegativeSafeInteger,
})
export type CatalogMissingComponent = typeof CatalogMissingComponentSchema.Type

/**
 * A catalog model on this computer. `drafterWithheld`: it runs without its catalog drafter, because it fits this
 * computer only without it, so it loads rather than being refused. `storedComponents`: the parts Magnitude stores
 * for a model whose weights are only in another app's location (its projector and drafter); Remove deletes these
 * and the model keeps running from the found file, and with none of them Remove is Hide.
 */
export const CatalogLocalModelSchema = Schema.TaggedStruct("Catalog", {
  ...LocalModelFields,
  modelId: CatalogFormModelIdSchema,
  servingState: CatalogLocalModelServingStateSchema,
  catalogData: LocalModelCatalogDataSchema,
  storageBytes: NonNegativeSafeInteger,
  acquisitionState: LocalModelAcquisitionStateSchema,
  missingComponents: Schema.Array(CatalogMissingComponentSchema),
  storedComponents: Schema.Array(CatalogComponentSchema),
  drafterWithheld: Schema.Boolean,
})
export type CatalogLocalModel = typeof CatalogLocalModelSchema.Type

export const CustomLocalModelSchema = Schema.TaggedStruct("Custom", {
  modelId: CustomFormModelIdSchema,
  origin: CustomModelOriginSchema,
  presentation: CustomModelPresentationSchema,
  drafter: Schema.optionalWith(AttachedDrafterSchema, { as: "Option", exact: true }),
  drafterChoice: DrafterChoiceSchema,
  recommendedDrafter: Schema.optionalWith(DrafterCandidateSchema, { as: "Option", exact: true }),
  acquisitionState: CustomModelAcquisitionStateSchema,
  servingState: CustomLocalModelServingStateSchema,
  storageBytes: NonNegativeSafeInteger,
}).pipe(Schema.filter((model) => {
  const form = modelIdForm(model.modelId)
  if (model.origin._tag === "Added") {
    return form === "HuggingFace" && isManagedAcquisitionState(model.acquisitionState)
  }
  if (isManagedAcquisitionState(model.acquisitionState)) return false
  if (model.origin._tag === "Found" && model.origin.source._tag === "HuggingFaceCache") return form === "HuggingFace"
  return true
}, {
  message: () => "added models carry a Hugging Face ID and managed acquisition; imported and found models are external",
}))
export type CustomLocalModel = typeof CustomLocalModelSchema.Type

export const LocalModelSchema = Schema.Union(CatalogLocalModelSchema, CustomLocalModelSchema)
export type LocalModel = typeof LocalModelSchema.Type

/**
 * A complete GGUF in a model location the engine cannot run; never callable, never a row. `modelId` is
 * its ID when it has one (a readable file the engine refuses), so a command naming it is refused with
 * these reasons.
 */
export const UnsupportedLocalModelSchema = Schema.Struct({
  modelId: Schema.optionalWith(ModelIdSchema, { as: "Option", exact: true }),
  displayName: NonEmptyString,
  origin: CustomModelOriginSchema,
  path: NonEmptyString,
  reasons: Schema.NonEmptyArray(UnsupportedReasonSchema),
  detail: Schema.String,
})
export type UnsupportedLocalModel = typeof UnsupportedLocalModelSchema.Type

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

/**
 * `hidden` lists found models the user hid: custom models, and catalog models whose files are only in
 * an external location. They are never in `models` and never offered.
 */
export const LocalModelsStateSchema = Schema.Struct({
  preparation: LocalModelPreparationSchema,
  models: Schema.Array(LocalModelSchema),
  unsupported: Schema.Array(UnsupportedLocalModelSchema),
  hidden: Schema.Array(LocalModelSchema),
}).pipe(Schema.filter(({ models, hidden }) => {
  const identities = [...models, ...hidden].map(({ modelId }) => modelId)
  return new Set(identities).size === identities.length
}, { message: () => "local models must have unique canonical model IDs" }))
export type LocalModelsState = typeof LocalModelsStateSchema.Type

export const HuggingFaceSearchResultSchema = Schema.Struct({
  repository: HuggingFaceRepositoryIdSchema,
  displayName: NonEmptyString,
  uploader: NonEmptyString,
  variantOf: Schema.optionalWith(CatalogFormModelIdSchema, { as: "Option", exact: true }),
  downloads: NonNegativeSafeInteger,
  architecture: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  unsupported: Schema.optionalWith(Schema.NonEmptyArray(UnsupportedReasonSchema), { as: "Option", exact: true }),
  gated: Schema.Boolean,
  versionCount: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
})
export type HuggingFaceSearchResult = typeof HuggingFaceSearchResultSchema.Type

/** Repositories of every architecture, unsupported ones last. */
export const HuggingFaceSearchResponseSchema = Schema.Struct({
  results: Schema.Array(HuggingFaceSearchResultSchema),
})
export type HuggingFaceSearchResponse = typeof HuggingFaceSearchResponseSchema.Type

export const CandidateSourceSchema = Schema.Union(
  Schema.TaggedStruct("HuggingFace", { repository: HuggingFaceRepositoryIdSchema, commit: NonEmptyString }),
  Schema.TaggedStruct("Local", { path: NonEmptyString }),
)
export type CandidateSource = typeof CandidateSourceSchema.Type

export const CandidateFileSchema = Schema.Struct({
  path: NonEmptyString,
  sizeBytes: NonNegativeSafeInteger,
})
export type CandidateFile = typeof CandidateFileSchema.Type

export const CandidateVersionSchema = Schema.Struct({
  label: NonEmptyString,
  modelId: CustomFormModelIdSchema,
  files: Schema.NonEmptyArray(NonEmptyString),
  sizeBytes: NonNegativeSafeInteger,
  downloadBytes: NonNegativeSafeInteger,
  assessment: Schema.optionalWith(LocalModelAssessmentSchema, { as: "Option", exact: true }),
  recommended: Schema.Boolean,
  catalogModel: Schema.optionalWith(CatalogFormModelIdSchema, { as: "Option", exact: true }),
  onThisComputer: Schema.optionalWith(ModelIdSchema, { as: "Option", exact: true }),
})
export type CandidateVersion = typeof CandidateVersionSchema.Type

const AddOutcomeSchema = Schema.Literal("Started", "AlreadyOnThisComputer")

/** What adding a version did: declared a custom model, or, for a version that is exactly a catalog model, installed that model. */
export const AddedModelSchema = Schema.Union(
  Schema.TaggedStruct("Custom", {
    modelId: CustomFormModelIdSchema,
    version: NonEmptyString,
    drafter: Schema.optionalWith(DrafterCandidateSchema, { as: "Option", exact: true }),
    outcome: AddOutcomeSchema,
    drafterDownloadStarted: Schema.Boolean,
  }),
  Schema.TaggedStruct("Catalog", {
    modelId: CatalogFormModelIdSchema,
    version: NonEmptyString,
    outcome: AddOutcomeSchema,
  }),
)
export type AddedModel = typeof AddedModelSchema.Type

/** The repository step: every version with the engine's verdict, before anything downloads. */
export const CandidateInspectionSchema = Schema.Struct({
  source: CandidateSourceSchema,
  displayName: NonEmptyString,
  license: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  variantOf: Schema.optionalWith(CatalogFormModelIdSchema, { as: "Option", exact: true }),
  versions: Schema.Array(CandidateVersionSchema),
  selectedVersion: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  projector: Schema.optionalWith(CandidateFileSchema, { as: "Option", exact: true }),
  unsupportedArchitecture: Schema.optionalWith(Schema.NonEmptyArray(UnsupportedReasonSchema), {
    as: "Option",
    exact: true,
  }),
}).pipe(Schema.filter((inspection) => Option.match(inspection.selectedVersion, {
  onNone: () => true,
  onSome: (label) => inspection.versions.some((version) => version.label === label),
}) && inspection.versions.filter(({ recommended }) => recommended).length <= 1, {
  message: () => "a candidate's selected version must be listed and at most one version is recommended",
}))
export type CandidateInspection = typeof CandidateInspectionSchema.Type

/** What an inspection is doing and how much of it is done; a drafter total counts the drafter checks known so far. */
export const CandidateInspectionPhaseSchema = Schema.Union(
  Schema.TaggedStruct("ReadingHeaders", { completed: NonNegativeSafeInteger, total: NonNegativeSafeInteger }),
  Schema.TaggedStruct("CheckingVersions", { completed: NonNegativeSafeInteger, total: NonNegativeSafeInteger }),
  Schema.TaggedStruct("CheckingDrafters", { version: NonEmptyString, completed: NonNegativeSafeInteger, total: NonNegativeSafeInteger }),
)
export type CandidateInspectionPhase = typeof CandidateInspectionPhaseSchema.Type

/** A version as an inspection in progress knows it: label, size, and its plain verdict once checked. */
export const CandidateVersionProgressSchema = Schema.Struct({
  label: NonEmptyString,
  sizeBytes: NonNegativeSafeInteger,
  assessment: Schema.optionalWith(LocalModelAssessmentSchema, { as: "Option", exact: true }),
})
export type CandidateVersionProgress = typeof CandidateVersionProgressSchema.Type

export const CandidateInspectionProgressSchema = Schema.Struct({
  phase: CandidateInspectionPhaseSchema,
  versions: Schema.Array(CandidateVersionProgressSchema),
})
export type CandidateInspectionProgress = typeof CandidateInspectionProgressSchema.Type

/** An inspection as it goes: its progress, in order, then the inspection itself. */
export const CandidateInspectionUpdateSchema = Schema.Union(
  Schema.TaggedStruct("Inspecting", { progress: CandidateInspectionProgressSchema }),
  Schema.TaggedStruct("Inspected", { inspection: CandidateInspectionSchema }),
)
export type CandidateInspectionUpdate = typeof CandidateInspectionUpdateSchema.Type

/** What a drafter list is for: an existing model, or a candidate's chosen files. */
export const DrafterTargetSchema = Schema.Union(
  Schema.TaggedStruct("Model", { modelId: ModelIdSchema }),
  Schema.TaggedStruct("Candidate", {
    source: CandidateSourceSchema,
    files: Schema.NonEmptyArray(NonEmptyString),
  }),
)
export type DrafterTarget = typeof DrafterTargetSchema.Type

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
  unsupportedLocalModels: Schema.Array(UnsupportedLocalModelSchema),
  hiddenLocalModels: Schema.Array(LocalModelSchema),
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
    case "Stopped":
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
  backend: LocalInferenceBackendSchema,
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
  physicalCores: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
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
