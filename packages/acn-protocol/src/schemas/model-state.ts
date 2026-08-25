import { Option, Schema } from "effect"
import {
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from "@magnitudedev/ai/provider/model"
import { FSM } from "@magnitudedev/utils"
import { ModelPackageInstallationOrigin as ModelPackageInstallationOriginSchema } from "@magnitudedev/icn-protocol/schemas"
export type { ModelPackageInstallationOrigin } from "@magnitudedev/icn-protocol/schemas"

const { defineFSM } = FSM

const NonNegativeSafeInteger = Schema.Number.pipe(
  Schema.int(),
  Schema.nonNegative(),
  Schema.lessThanOrEqualTo(Number.MAX_SAFE_INTEGER),
)
const PositiveSafeInteger = NonNegativeSafeInteger.pipe(Schema.positive())
const FiniteNonNegative = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const NonEmptyString = Schema.String.pipe(Schema.minLength(1))
const Sha256Digest = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))

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
  code: Schema.String,
  message: Schema.String,
  retryable: Schema.Boolean,
})
export type ModelFailure = typeof ModelFailureSchema.Type

export const ModelDownloadFailureSchema = Schema.Union(
  Schema.TaggedStruct("Interrupted", {}),
  Schema.TaggedStruct("InsufficientDiskSpace", {
    requiredBytes: NonNegativeSafeInteger,
    availableBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("SourceUnavailable", {}),
  Schema.TaggedStruct("NetworkUnavailable", {}),
  Schema.TaggedStruct("LocalStorageFailure", {}),
  Schema.TaggedStruct("CorruptDownload", {}),
  Schema.TaggedStruct("Internal", { message: Schema.String }),
)
export type ModelDownloadFailure = typeof ModelDownloadFailureSchema.Type

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
// Local model packages, bundles, assessments, and offerings
// =============================================================================

export const ModelFileIdSchema = NonEmptyString.pipe(Schema.brand("ModelFileId"))
export type ModelFileId = typeof ModelFileIdSchema.Type

export const ModelPackageIdSchema = NonEmptyString.pipe(Schema.brand("ModelPackageId"))
export type ModelPackageId = typeof ModelPackageIdSchema.Type

export const ModelDownloadIdSchema = NonEmptyString.pipe(Schema.brand("ModelDownloadId"))
export type ModelDownloadId = typeof ModelDownloadIdSchema.Type

export const ModelDownloadStageSchema = Schema.Literal(
  "queued",
  "resolving",
  "checking_space",
  "downloading",
  "verifying",
  "publishing",
)
export type ModelDownloadStage = typeof ModelDownloadStageSchema.Type

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

export const CatalogModelIdSchema = NonEmptyString.pipe(
  Schema.filter((value) => !value.includes(":"), {
    message: () => "catalog model ID must contain one identity component",
  }),
  Schema.brand("CatalogModelId"),
)
export type CatalogModelId = typeof CatalogModelIdSchema.Type

export const CatalogVariantIdSchema = NonEmptyString.pipe(
  Schema.filter((value) => {
    const components = value.split(":")
    return components.length === 2 && components.every((component) => component.length > 0)
  }, { message: () => "catalog variant ID must be format:quality" }),
  Schema.brand("CatalogVariantId"),
)
export type CatalogVariantId = typeof CatalogVariantIdSchema.Type

export const CatalogIdentitySchema = Schema.Struct({
  modelId: CatalogModelIdSchema,
  variantId: CatalogVariantIdSchema,
})
export type CatalogIdentity = typeof CatalogIdentitySchema.Type

export const RecommendationIdSchema = NonEmptyString.pipe(Schema.brand("RecommendationId"))
export type RecommendationId = typeof RecommendationIdSchema.Type

export const ModelAssessmentIdSchema =
  NonEmptyString.pipe(Schema.brand("ModelAssessmentId"))
export type ModelAssessmentId = typeof ModelAssessmentIdSchema.Type

export const AssessmentEnvironmentIdSchema =
  NonEmptyString.pipe(Schema.brand("AssessmentEnvironmentId"))
export type AssessmentEnvironmentId = typeof AssessmentEnvironmentIdSchema.Type

export const ModelFileRoleSchema = Schema.Literal(
  "weights",
  "projector",
  "draft",
  "mtp",
  "auxiliary",
)
export type ModelFileRole = typeof ModelFileRoleSchema.Type

export const ModelFileSchema = Schema.Struct({
  id: ModelFileIdSchema,
  path: NonEmptyString,
  role: ModelFileRoleSchema,
  sizeBytes: NonNegativeSafeInteger,
  tensorStorageBytes: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  sha256: Sha256Digest,
})
export type ModelFile = typeof ModelFileSchema.Type

export const ModelPackageSourceSchema = Schema.Union(
  Schema.TaggedStruct("HuggingFace", {
    repository: NonEmptyString,
    revision: NonEmptyString,
  }),
  Schema.TaggedStruct("Local", {
    path: NonEmptyString,
  }),
)
export type ModelPackageSource = typeof ModelPackageSourceSchema.Type

export const SpeculativeMethodSchema = Schema.Union(
  Schema.TaggedStruct("Mtp", {}),
  Schema.TaggedStruct("DFlash", {}),
  Schema.TaggedStruct("DSpark", {}),
)
export type SpeculativeMethod = typeof SpeculativeMethodSchema.Type

export const ModelFileRelationshipSchema = Schema.Union(
  Schema.TaggedStruct("Shard", {
    fileId: ModelFileIdSchema,
    index: NonNegativeSafeInteger,
    count: PositiveSafeInteger,
  }),
  Schema.TaggedStruct("ProjectorFor", {
    projectorFileId: ModelFileIdSchema,
    weightsFileId: ModelFileIdSchema,
  }),
  Schema.TaggedStruct("MtpFor", {
    mtpFileId: ModelFileIdSchema,
    weightsFileId: ModelFileIdSchema,
  }),
  Schema.TaggedStruct("DraftFor", {
    draftFileId: ModelFileIdSchema,
    weightsFileId: ModelFileIdSchema,
    method: SpeculativeMethodSchema,
  }),
)
export type ModelFileRelationship = typeof ModelFileRelationshipSchema.Type

export const ModelPackagePropertiesSchema = Schema.Struct({
  format: NonEmptyString,
  quantization: NonEmptyString,
  quantizationName: NonEmptyString,
  architecture: NonEmptyString,
  maximumContextLength: Schema.optionalWith(PositiveSafeInteger, { as: "Option", exact: true }),
  intrinsicModelId: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
  intrinsicQualityId: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
})
export type ModelPackageProperties = typeof ModelPackagePropertiesSchema.Type

export const ModelPackageSchema = Schema.Struct({
  id: ModelPackageIdSchema,
  source: ModelPackageSourceSchema,
  files: Schema.Array(ModelFileSchema),
  relationships: Schema.Array(ModelFileRelationshipSchema),
  properties: ModelPackagePropertiesSchema,
})
export type ModelPackage = typeof ModelPackageSchema.Type

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

export const ModelPackageInspectionSchema = Schema.Union(
  Schema.TaggedStruct("Pending", {}),
  Schema.TaggedStruct("Inspected", { capabilities: ModelCapabilitiesSchema }),
  Schema.TaggedStruct("Invalid", { failure: ModelFailureSchema }),
  Schema.TaggedStruct("Incompatible", { failure: ModelFailureSchema }),
)
export type ModelPackageInspection = typeof ModelPackageInspectionSchema.Type

export const ModelPackageLocalStateSchema = Schema.Union(
  Schema.TaggedStruct("NotInstalled", {}),
  Schema.TaggedStruct("Installed", {
    path: NonEmptyString,
    origin: ModelPackageInstallationOriginSchema,
  }),
)
export type ModelPackageLocalState = typeof ModelPackageLocalStateSchema.Type

export const InstalledCatalogAttributionSchema = Schema.Union(
  Schema.TaggedStruct("NotCatalogTarget", {}),
  Schema.TaggedStruct("Attributed", CatalogIdentitySchema.fields),
  Schema.TaggedStruct("Failed", { failure: ModelFailureSchema }),
)
export type InstalledCatalogAttribution = typeof InstalledCatalogAttributionSchema.Type

export const ModelPackageEntrySchema = Schema.Struct({
  package: ModelPackageSchema,
  localState: ModelPackageLocalStateSchema,
  inspection: ModelPackageInspectionSchema,
  catalogAttribution: InstalledCatalogAttributionSchema,
})
export type ModelPackageEntry = typeof ModelPackageEntrySchema.Type

export const StandaloneModelBundleSchema = Schema.TaggedStruct("Standalone", {
  package: ModelPackageSchema,
})

export const SpeculativeDraftSourceSchema = Schema.Union(
  Schema.TaggedStruct("Embedded", {}),
  Schema.TaggedStruct("Separate", {
    draft: ModelPackageSchema,
  }),
)
export type SpeculativeDraftSource = typeof SpeculativeDraftSourceSchema.Type

export const SpeculativeDecodingModelBundleSchema = Schema.TaggedStruct("SpeculativeDecoding", {
  target: ModelPackageSchema,
  draftSource: SpeculativeDraftSourceSchema,
  method: SpeculativeMethodSchema,
})

export const ServableModelBundleSchema = Schema.Union(
  StandaloneModelBundleSchema,
  SpeculativeDecodingModelBundleSchema,
)
export type ServableModelBundle = typeof ServableModelBundleSchema.Type

export const ModelBundleDownloadStateSchema = Schema.Union(
  Schema.TaggedStruct("Pending", {
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("Downloading", {
    stage: ModelDownloadStageSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Completed", {}),
  Schema.TaggedStruct("Failed", {
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    failure: ModelDownloadFailureSchema,
    acknowledged: Schema.Boolean,
  }),
  Schema.TaggedStruct("Cancelled", {
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
  }),
)
export type ModelBundleDownloadState = typeof ModelBundleDownloadStateSchema.Type

export const ModelBundleDownloadSchema = Schema.Struct({
  id: ModelDownloadIdSchema,
  bundle: ServableModelBundleSchema,
  state: ModelBundleDownloadStateSchema,
})
export type ModelBundleDownload = typeof ModelBundleDownloadSchema.Type

export const servableModelBundlePackages = (
  bundle: ServableModelBundle,
): readonly ModelPackage[] => bundle._tag === "Standalone"
  ? [bundle.package]
  : bundle.draftSource._tag === "Embedded"
    ? [bundle.target]
    : [bundle.target, bundle.draftSource.draft]

export const servableModelBundlePackageIds = (
  bundle: ServableModelBundle,
): readonly ModelPackageId[] => servableModelBundlePackages(bundle).map(({ id }) => id)

export const servableModelBundleTargetPackageId = (
  bundle: ServableModelBundle,
): ModelPackageId => bundle._tag === "Standalone" ? bundle.package.id : bundle.target.id

export const sameServableModelBundleIdentity = (
  left: ServableModelBundle,
  right: ServableModelBundle,
): boolean => {
  if (left._tag !== right._tag) return false
  if (left._tag === "SpeculativeDecoding" && right._tag === "SpeculativeDecoding") {
    if (left.draftSource._tag !== right.draftSource._tag) return false
    if (!Schema.equivalence(SpeculativeMethodSchema)(left.method, right.method)) return false
  }
  const leftPackageIds = servableModelBundlePackageIds(left)
  const rightPackageIds = servableModelBundlePackageIds(right)
  return leftPackageIds.length === rightPackageIds.length
    && leftPackageIds.every((packageId, index) => rightPackageIds[index] === packageId)
}

export const ServingProfileSchema = Schema.Struct({
  contextLength: PositiveSafeInteger,
})
export type ServingProfile = typeof ServingProfileSchema.Type

export const ModelServingConfigurationSchema = Schema.Struct({
  bundle: ServableModelBundleSchema,
  profile: ServingProfileSchema,
})
export type ModelServingConfiguration = typeof ModelServingConfigurationSchema.Type

export const RecommendableModelCapabilitiesSchema = ModelCapabilitiesSchema
export type RecommendableModelCapabilities = typeof RecommendableModelCapabilitiesSchema.Type

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

export const RecommendableModelSchema = Schema.Struct({
  ...CatalogIdentitySchema.fields,
  configuration: ModelServingConfigurationSchema,
  displayName: NonEmptyString,
  variantLabel: ModelVariantLabelSchema,
  description: Schema.String,
  releaseDate: ModelReleaseDateSchema,
  license: NonEmptyString,
  capabilities: RecommendableModelCapabilitiesSchema,
  parameterization: ModelParameterizationSchema,
  qualityScore: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
  qualityScoreProvenance: NonEmptyString,
  fidelityRank: NonNegativeSafeInteger,
  quantizationAware: Schema.Boolean,
  qualityEvidence: Schema.Array(NonEmptyString),
})
export type RecommendableModel = typeof RecommendableModelSchema.Type

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

export const FitsModelAssessmentSchema = Schema.TaggedStruct("Fits", {
  profile: ServingProfileSchema,
  assessmentId: ModelAssessmentIdSchema,
  environmentId: AssessmentEnvironmentIdSchema,
  memory: Schema.Array(MemoryAssessmentSchema),
  performance: GenerationPerformanceSamplesSchema,
}).pipe(Schema.filter((assessment) =>
  assessment.performance.at(-1)?.contextTokens === assessment.profile.contextLength,
{ message: () => "performance samples must end at the serving profile context" }))
export type FitsModelAssessment = typeof FitsModelAssessmentSchema.Type

export const LocalModelInstalledPackageSchema = Schema.Struct({
  packageId: ModelPackageIdSchema,
  path: NonEmptyString,
  origin: ModelPackageInstallationOriginSchema,
})
export type LocalModelInstalledPackage = typeof LocalModelInstalledPackageSchema.Type

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
  stage: ModelDownloadStageSchema,
  completedBytes: NonNegativeSafeInteger,
  totalBytes: NonNegativeSafeInteger,
  bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
})
export type ModelTransferProgress = typeof ModelTransferProgressSchema.Type

const InstalledModelFields = {
  installedBytes: NonNegativeSafeInteger,
  packages: Schema.NonEmptyArray(LocalModelInstalledPackageSchema),
  residencyState: ModelResidencySchema,
} as const

const uniqueInstalledPackages = <A extends {
  readonly packages: readonly { readonly packageId: string }[]
}, I>(schema: Schema.Schema<A, I>) => schema.pipe(Schema.filter(({ packages }) =>
  new Set(packages.map(({ packageId }) => packageId)).size === packages.length,
{ message: () => "installed local model packages must have unique package identities" }))

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
  Schema.TaggedStruct("InstallFailed", { failure: ModelDownloadFailureSchema }),
  uniqueInstalledPackages(Schema.TaggedStruct("Installed", InstalledModelFields)),
  uniqueInstalledPackages(Schema.TaggedStruct("UpdateAvailable", InstalledModelFields)),
  uniqueInstalledPackages(Schema.TaggedStruct("Updating", {
    ...InstalledModelFields,
    progress: ModelTransferProgressSchema,
  })),
  /** Acknowledged failure returns to UpdateAvailable. */
  uniqueInstalledPackages(Schema.TaggedStruct("UpdateFailed", {
    ...InstalledModelFields,
    failure: ModelDownloadFailureSchema,
  })),
)
export type LocalModelAcquisitionState = typeof LocalModelAcquisitionStateSchema.Type

export type InstalledLocalModelAcquisitionState = Extract<
  LocalModelAcquisitionState,
  { readonly packages: unknown }
>

/** The installed-family payload when any version of the model is on disk. */
export const installedAcquisition = (
  state: LocalModelAcquisitionState,
): InstalledLocalModelAcquisitionState | undefined => state._tag === "Installed"
  || state._tag === "UpdateAvailable"
  || state._tag === "Updating"
  || state._tag === "UpdateFailed"
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
): ModelDownloadFailure | undefined => state._tag === "InstallFailed" || state._tag === "UpdateFailed"
  ? state.failure
  : undefined

export const CatalogModelReconciliationAdmissionSchema = Schema.Union(
  Schema.TaggedStruct("Current", {
    providerModelId: ProviderModelIdSchema,
  }),
  Schema.TaggedStruct("DownloadAdmitted", {
    providerModelId: ProviderModelIdSchema,
  }),
)
export type CatalogModelReconciliationAdmission =
  typeof CatalogModelReconciliationAdmissionSchema.Type

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

export const LocalModelConfigurationAssessmentSchema = Schema.Union(
  Schema.TaggedStruct("Failed", { failure: ModelFailureSchema }),
  Schema.TaggedStruct("Fits", {
    assessment: FitsModelAssessmentSchema,
  }),
  Schema.TaggedStruct("DoesNotFit", {
    assessmentId: ModelAssessmentIdSchema,
    environmentId: AssessmentEnvironmentIdSchema,
    memory: Schema.Array(MemoryAssessmentSchema),
    totalRequiredBytes: NonNegativeSafeInteger,
    deficitBytes: NonNegativeSafeInteger,
    limitingResource: NonEmptyString,
  }),
  Schema.TaggedStruct("Incompatible", {
    environmentId: AssessmentEnvironmentIdSchema,
    failure: ModelFailureSchema,
  }),
)
export type LocalModelConfigurationAssessment =
  typeof LocalModelConfigurationAssessmentSchema.Type

export const LocalModelCatalogDataSchema = Schema.Struct({
  ...CatalogIdentitySchema.fields,
  releaseDate: ModelReleaseDateSchema,
  parameterization: ModelParameterizationSchema,
  intelligenceScore: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
  intelligenceScoreSource: NonEmptyString,
  fidelityRank: NonNegativeSafeInteger,
  quantizationAware: Schema.Boolean,
  qualityNotes: Schema.Array(NonEmptyString),
})
export type LocalModelCatalogData = typeof LocalModelCatalogDataSchema.Type

export const LocalModelCatalogMembershipStateSchema = Schema.Union(
  Schema.TaggedStruct("NotInCatalog", {}),
  Schema.TaggedStruct("AttributionFailed", { failure: ModelFailureSchema }),
  Schema.TaggedStruct("InCatalog", {
    catalogData: LocalModelCatalogDataSchema,
  }),
)
export type LocalModelCatalogMembershipState =
  typeof LocalModelCatalogMembershipStateSchema.Type

export const LocalModelRecommendationSchema = Schema.Struct({
  id: RecommendationIdSchema,
  intent: Schema.Literal("balanced", "smartest", "fastest", "lightweight"),
  explanation: Schema.String,
})
export type LocalModelRecommendation = typeof LocalModelRecommendationSchema.Type

export const LocalModelMemoryHeadroomObservationSchema = Schema.Struct({
  requiredSystemMemoryBytes: NonNegativeSafeInteger,
  allocationHeadroomBytes: NonNegativeSafeInteger,
  abortReserveBytes: NonNegativeSafeInteger,
  loadBoundaryBytes: NonNegativeSafeInteger,
})
export type LocalModelMemoryHeadroomObservation =
  typeof LocalModelMemoryHeadroomObservationSchema.Type

export const LocalModelSystemMemoryUseStateSchema = Schema.Union(
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
})
export type LocalModelMemory = typeof LocalModelMemorySchema.Type

export const LocalModelAssessmentSchema = Schema.Union(
  Schema.TaggedStruct("Fits", {
    assessmentId: ModelAssessmentIdSchema,
    environmentId: AssessmentEnvironmentIdSchema,
    profile: ServingProfileSchema,
    memory: LocalModelMemorySchema,
    performance: GenerationPerformanceSamplesSchema,
  }),
  Schema.TaggedStruct("DoesNotFit", {
    assessmentId: ModelAssessmentIdSchema,
    environmentId: AssessmentEnvironmentIdSchema,
    memoryDomains: Schema.Array(MemoryAssessmentSchema),
    totalRequiredBytes: NonNegativeSafeInteger,
    deficitBytes: NonNegativeSafeInteger,
    limitingResource: NonEmptyString,
  }),
  Schema.TaggedStruct("Incompatible", {
    environmentId: AssessmentEnvironmentIdSchema,
    failure: ModelFailureSchema,
  }),
)
export type LocalModelAssessment = typeof LocalModelAssessmentSchema.Type

export const LocalModelAvailabilityStateSchema = Schema.Union(
  Schema.TaggedStruct("Installable", {}),
  Schema.TaggedStruct("Preparing", { providerModelId: ProviderModelIdSchema }),
  Schema.TaggedStruct("Selectable", { providerModelId: ProviderModelIdSchema }),
  Schema.TaggedStruct("Unavailable", {
    providerModelId: Schema.optionalWith(ProviderModelIdSchema, { as: "Option", exact: true }),
    failure: ModelFailureSchema,
  }),
)
export type LocalModelAvailabilityState = typeof LocalModelAvailabilityStateSchema.Type

export const LocalModelPresentationSchema = Schema.Struct({
  displayName: NonEmptyString,
  variantLabel: ModelVariantLabelSchema,
  description: Schema.String,
  license: Schema.optionalWith(NonEmptyString, { as: "Option", exact: true }),
})
export type LocalModelPresentation = typeof LocalModelPresentationSchema.Type

export const LocalModelServingStateSchema = Schema.Union(
  Schema.TaggedStruct("Resolving", {}),
  Schema.TaggedStruct("Assessing", {
    configuration: ModelServingConfigurationSchema,
  }),
  Schema.TaggedStruct("Failed", {
    configuration: Schema.optionalWith(ModelServingConfigurationSchema, {
      as: "Option",
      exact: true,
    }),
    failure: ModelFailureSchema,
  }),
  Schema.TaggedStruct("Assessed", {
    configuration: ModelServingConfigurationSchema,
    capabilities: ModelCapabilitiesSchema,
    assessment: LocalModelAssessmentSchema,
    availabilityState: LocalModelAvailabilityStateSchema,
    recommendations: Schema.Array(LocalModelRecommendationSchema),
  }),
)
export type LocalModelServingState = typeof LocalModelServingStateSchema.Type

export const LocalModelSchema = Schema.Struct({
  modelId: ProviderModelIdSchema,
  bundle: ServableModelBundleSchema,
  presentation: LocalModelPresentationSchema,
  downloadBytes: NonNegativeSafeInteger,
  catalogMembershipState: LocalModelCatalogMembershipStateSchema,
  acquisitionState: LocalModelAcquisitionStateSchema,
  servingState: LocalModelServingStateSchema,
}).pipe(Schema.filter((model) => {
  // Only the current target's installation matches the row's bundle exactly;
  // the update-family variants may hold a prior version's packages.
  if (model.acquisitionState._tag !== "Installed") return true
  const bundlePackageIds = servableModelBundlePackageIds(model.bundle)
  const installedPackageIds = model.acquisitionState.packages.map(({ packageId }) => packageId)
  return bundlePackageIds.length === installedPackageIds.length
    && bundlePackageIds.every((packageId) => installedPackageIds.includes(packageId))
}, { message: () => "installed local models must carry the exact location of every bundle package" }))
export type LocalModel = typeof LocalModelSchema.Type

export const LocalModelRecommendationProgressStepIdSchema = Schema.Literal(
  "hardware",
  "inventory",
  "assessment",
  "recommendations",
)
export type LocalModelRecommendationProgressStepId =
  typeof LocalModelRecommendationProgressStepIdSchema.Type

export const LocalModelRecommendationProgressStatusSchema = Schema.Union(
  Schema.TaggedStruct("Pending", {}),
  Schema.TaggedStruct("Running", {
    startedAtMs: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("Completed", {
    startedAtMs: NonNegativeSafeInteger,
    durationMs: NonNegativeSafeInteger,
    cached: Schema.Boolean,
  }),
  Schema.TaggedStruct("Failed", {
    startedAtMs: NonNegativeSafeInteger,
    durationMs: NonNegativeSafeInteger,
    failure: ModelFailureSchema,
  }),
)
export type LocalModelRecommendationProgressStatus =
  typeof LocalModelRecommendationProgressStatusSchema.Type

export const LocalModelRecommendationProgressStepSchema = Schema.Struct({
  id: LocalModelRecommendationProgressStepIdSchema,
  status: LocalModelRecommendationProgressStatusSchema,
  completedItems: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  totalItems: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  estimatedRemainingMs: Schema.optionalWith(
    NonNegativeSafeInteger,
    { as: "Option", exact: true },
  ),
})
export type LocalModelRecommendationProgressStep =
  typeof LocalModelRecommendationProgressStepSchema.Type

export const ModelRecommendationSchema = Schema.Struct({
  modelId: ProviderModelIdSchema,
  id: RecommendationIdSchema,
  intent: Schema.Literal("balanced", "smartest", "fastest", "lightweight"),
  explanation: Schema.String,
})
export type ModelRecommendation = typeof ModelRecommendationSchema.Type

export const ModelRecommendationsStateSchema = Schema.Union(
  Schema.TaggedStruct("Loading", {
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Ready", {
    recommendations: Schema.Array(ModelRecommendationSchema),
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Failed", {
    failure: ModelFailureSchema,
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
)
export type ModelRecommendationsState = typeof ModelRecommendationsStateSchema.Type

export const LocalModelDiscoveryStateSchema = Schema.Union(
  Schema.TaggedStruct("Loading", {
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Ready", {
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Failed", {
    failure: ModelFailureSchema,
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
)
export type LocalModelDiscoveryState = typeof LocalModelDiscoveryStateSchema.Type

export const LocalModelInventoryStateSchema = Schema.Union(
  Schema.TaggedStruct("Initializing", {}),
  Schema.TaggedStruct("Ready", {}),
  Schema.TaggedStruct("Degraded", { failure: ModelFailureSchema }),
)
export type LocalModelInventoryState = typeof LocalModelInventoryStateSchema.Type

export const LocalModelsStateSchema = Schema.Struct({
  inventoryState: LocalModelInventoryStateSchema,
  models: Schema.Array(LocalModelSchema),
  discoveryState: LocalModelDiscoveryStateSchema,
}).pipe(Schema.filter(({ models }) => {
  const identities = models.map(({ bundle }) => servableModelBundleTargetPackageId(bundle))
  return new Set(identities).size === identities.length
}, { message: () => "local models must have unique target-package identities" }))
export type LocalModelsState = typeof LocalModelsStateSchema.Type

export const ModelPackagesStateSchema = Schema.Struct({
  inventory: Schema.Union(
    Schema.TaggedStruct("Initializing", {}),
    Schema.TaggedStruct("Ready", {}),
    Schema.TaggedStruct("Degraded", { failure: ModelFailureSchema }),
  ),
  entries: Schema.Array(ModelPackageEntrySchema),
  downloads: Schema.Array(ModelBundleDownloadSchema),
})
export type ModelPackagesState = typeof ModelPackagesStateSchema.Type

export const LocalProviderOfferingSchema = Schema.Struct({
  providerModelId: ProviderModelIdSchema,
  configuration: ModelServingConfigurationSchema,
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
  localInventoryState: LocalModelInventoryStateSchema,
  localDiscoveryState: LocalModelDiscoveryStateSchema,
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
