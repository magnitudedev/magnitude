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
const Sha256Digest = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/))

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

// =============================================================================
// Local model packages, targets, assessments, and offerings
// =============================================================================

export const ModelFileIdSchema = NonEmptyString.pipe(Schema.brand("ModelFileId"))
export type ModelFileId = typeof ModelFileIdSchema.Type

export const ModelPackageIdSchema = NonEmptyString.pipe(Schema.brand("ModelPackageId"))
export type ModelPackageId = typeof ModelPackageIdSchema.Type

export const DownloadAttemptIdSchema = NonEmptyString.pipe(Schema.brand("DownloadAttemptId"))
export type DownloadAttemptId = typeof DownloadAttemptIdSchema.Type

export const ModelDownloadStageSchema = Schema.Literal(
  "queued",
  "resolving",
  "checking_space",
  "downloading",
  "verifying",
  "publishing",
)
export type ModelDownloadStage = typeof ModelDownloadStageSchema.Type

export const ModelInstanceIdSchema = NonEmptyString.pipe(Schema.brand("ModelInstanceId"))
export type ModelInstanceId = typeof ModelInstanceIdSchema.Type

export const SpeculativeDecodingPairIdSchema =
  NonEmptyString.pipe(Schema.brand("SpeculativeDecodingPairId"))
export type SpeculativeDecodingPairId = typeof SpeculativeDecodingPairIdSchema.Type

export const ModelOfferingTargetIdSchema =
  NonEmptyString.pipe(Schema.brand("ModelOfferingTargetId"))
export type ModelOfferingTargetId = typeof ModelOfferingTargetIdSchema.Type

export const ModelServingConfigurationIdSchema =
  NonEmptyString.pipe(Schema.brand("ModelServingConfigurationId"))
export type ModelServingConfigurationId = typeof ModelServingConfigurationIdSchema.Type

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

export const RecommendableModelIdSchema =
  NonEmptyString.pipe(Schema.brand("RecommendableModelId"))
export type RecommendableModelId = typeof RecommendableModelIdSchema.Type

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
  Schema.TaggedStruct("DraftSimple", {}),
  Schema.TaggedStruct("DraftEagle3", {}),
  Schema.TaggedStruct("DraftDFlash", {}),
  Schema.TaggedStruct("Ngram", {
    method: NonEmptyString,
  }),
  Schema.TaggedStruct("UnknownNative", {
    method: NonEmptyString,
    evidence: NonEmptyString,
  }),
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
  maximumContextLength: PositiveSafeInteger,
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
  Schema.TaggedStruct("Downloading", {
    attemptId: DownloadAttemptIdSchema,
    stage: ModelDownloadStageSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("DownloadFailed", {
    attemptId: DownloadAttemptIdSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    failure: ModelFailureSchema,
  }),
  Schema.TaggedStruct("DownloadCancelled", {
    attemptId: DownloadAttemptIdSchema,
  }),
  Schema.TaggedStruct("Installed", { path: NonEmptyString }),
)
export type ModelPackageLocalState = typeof ModelPackageLocalStateSchema.Type

export const ModelPackageEntrySchema = Schema.Struct({
  package: ModelPackageSchema,
  targetId: Schema.optionalWith(ModelOfferingTargetIdSchema, { as: "Option", exact: true }),
  localState: ModelPackageLocalStateSchema,
  inspection: ModelPackageInspectionSchema,
})
export type ModelPackageEntry = typeof ModelPackageEntrySchema.Type

export const DownloadAttemptSchema = Schema.Union(
  Schema.TaggedStruct("Pending", {
    id: DownloadAttemptIdSchema,
    packageId: ModelPackageIdSchema,
  }),
  Schema.TaggedStruct("Downloading", {
    id: DownloadAttemptIdSchema,
    packageId: ModelPackageIdSchema,
    stage: ModelDownloadStageSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Completed", {
    id: DownloadAttemptIdSchema,
    packageId: ModelPackageIdSchema,
  }),
  Schema.TaggedStruct("Failed", {
    id: DownloadAttemptIdSchema,
    packageId: ModelPackageIdSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    failure: ModelFailureSchema,
  }),
  Schema.TaggedStruct("Cancelled", {
    id: DownloadAttemptIdSchema,
    packageId: ModelPackageIdSchema,
  }),
)
export type DownloadAttempt = typeof DownloadAttemptSchema.Type

export const PackageOfferingTargetSchema = Schema.TaggedStruct("Package", {
  package: ModelPackageSchema,
})

export const SpeculativeDecodingPairSchema = Schema.TaggedStruct("SpeculativeDecodingPair", {
  id: SpeculativeDecodingPairIdSchema,
  target: ModelPackageSchema,
  draft: ModelPackageSchema,
})

export const ModelOfferingTargetSchema = Schema.Union(
  PackageOfferingTargetSchema,
  SpeculativeDecodingPairSchema,
)
export type ModelOfferingTarget = typeof ModelOfferingTargetSchema.Type

export const modelOfferingTargetPackageIds = (
  target: ModelOfferingTarget,
): readonly ModelPackageId[] =>
  target._tag === "Package"
    ? [target.package.id]
    : [target.target.id, target.draft.id]

export const ServingProfileSchema = Schema.Struct({
  contextLength: PositiveSafeInteger,
})
export type ServingProfile = typeof ServingProfileSchema.Type

export const ModelServingConfigurationSchema = Schema.Struct({
  id: ModelServingConfigurationIdSchema,
  target: ModelOfferingTargetSchema,
  profile: ServingProfileSchema,
})
export type ModelServingConfiguration = typeof ModelServingConfigurationSchema.Type

export const RecommendableModelCapabilitiesSchema = ModelCapabilitiesSchema
export type RecommendableModelCapabilities = typeof RecommendableModelCapabilitiesSchema.Type

export const RecommendableModelSchema = Schema.Struct({
  id: RecommendableModelIdSchema,
  checkpointId: NonEmptyString,
  targetId: ModelOfferingTargetIdSchema,
  target: ModelOfferingTargetSchema,
  profile: ServingProfileSchema,
  displayName: NonEmptyString,
  description: Schema.String,
  license: NonEmptyString,
  capabilities: RecommendableModelCapabilitiesSchema,
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
  warningReserveBytes: NonNegativeSafeInteger,
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
  configurationId: ModelServingConfigurationIdSchema,
  assessmentId: ModelAssessmentIdSchema,
  environmentId: AssessmentEnvironmentIdSchema,
  memory: Schema.Array(MemoryAssessmentSchema),
  performance: GenerationPerformanceSamplesSchema,
}).pipe(Schema.filter((assessment) =>
  assessment.performance.at(-1)?.contextTokens === assessment.profile.contextLength,
{ message: () => "performance samples must end at the serving profile context" }))
export type FitsModelAssessment = typeof FitsModelAssessmentSchema.Type

export const RecommendationSchema = Schema.Struct({
  id: RecommendationIdSchema,
  targetId: ModelOfferingTargetIdSchema,
  recommendableModelId: RecommendableModelIdSchema,
  displayName: NonEmptyString,
  description: Schema.String,
  configuration: ModelServingConfigurationSchema,
  assessment: FitsModelAssessmentSchema,
  intent: Schema.Literal("balanced", "best_quality", "fastest", "lightweight"),
  explanation: Schema.String,
})
export type Recommendation = typeof RecommendationSchema.Type

export const LocalModelDownloadSchema = Schema.Union(
  Schema.TaggedStruct("NotDownloaded", {
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("Downloading", {
    attemptIds: Schema.NonEmptyArray(DownloadAttemptIdSchema),
    stage: ModelDownloadStageSchema,
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    bytesPerSecond: Schema.optionalWith(NonNegativeSafeInteger, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Failed", {
    attemptIds: Schema.NonEmptyArray(DownloadAttemptIdSchema),
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
    failure: ModelFailureSchema,
  }),
  Schema.TaggedStruct("Cancelled", {
    attemptIds: Schema.NonEmptyArray(DownloadAttemptIdSchema),
    completedBytes: NonNegativeSafeInteger,
    totalBytes: NonNegativeSafeInteger,
  }),
  Schema.TaggedStruct("Downloaded", {
    installedBytes: NonNegativeSafeInteger,
  }),
)
export type LocalModelDownload = typeof LocalModelDownloadSchema.Type

export const ModelDownloadAdmissionSchema = Schema.Union(
  Schema.TaggedStruct("AlreadyInstalled", {}),
  Schema.TaggedStruct("DownloadAdmitted", {
    attemptIds: Schema.NonEmptyArray(DownloadAttemptIdSchema),
  }),
)
export type ModelDownloadAdmission = typeof ModelDownloadAdmissionSchema.Type

export const LocalModelCatalogCandidateAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("NotDownloaded", {}),
  Schema.TaggedStruct("Unavailable", {
    failure: ModelFailureSchema,
  }),
  Schema.TaggedStruct("Available", {}),
)
export type LocalModelCatalogCandidateAvailability =
  typeof LocalModelCatalogCandidateAvailabilitySchema.Type

export const LocalModelOfferingReferenceSchema = Schema.Struct({
  configurationId: ModelServingConfigurationIdSchema,
  providerModelId: ProviderModelIdSchema,
})
export type LocalModelOfferingReference = typeof LocalModelOfferingReferenceSchema.Type

export const LocalModelAssessmentLifecycleSchema = Schema.Union(
  Schema.TaggedStruct("Unassessed", {}),
  Schema.TaggedStruct("Assessing", {}),
  Schema.TaggedStruct("Assessed", {
    environmentId: AssessmentEnvironmentIdSchema,
    configurationIds: Schema.Array(ModelServingConfigurationIdSchema),
  }),
)
export type LocalModelAssessmentLifecycle = typeof LocalModelAssessmentLifecycleSchema.Type

export const LocalModelSchema = Schema.Struct({
  targetId: ModelOfferingTargetIdSchema,
  offerings: Schema.Array(LocalModelOfferingReferenceSchema),
  displayName: NonEmptyString,
  description: Schema.String,
  kind: Schema.Literal("Standalone", "SpeculativePair"),
  quantization: NonEmptyString,
  maximumContextLength: PositiveSafeInteger,
  downloadBytes: NonNegativeSafeInteger,
  download: LocalModelDownloadSchema,
  assessment: LocalModelAssessmentLifecycleSchema,
})
export type LocalModel = typeof LocalModelSchema.Type

export const LocalModelCatalogRecommendationEvidenceSchema = Schema.Struct({
  intelligence: Schema.optionalWith(Schema.Struct({
    score: Schema.Number.pipe(Schema.finite(), Schema.nonNegative()),
    provenance: NonEmptyString,
  }), { as: "Option", exact: true }),
  fidelityRank: NonNegativeSafeInteger,
  qualityEvidence: Schema.Array(NonEmptyString),
})
export type LocalModelCatalogRecommendationEvidence =
  typeof LocalModelCatalogRecommendationEvidenceSchema.Type

const LocalModelCatalogCandidateMetadataFields = {
  configurationId: ModelServingConfigurationIdSchema,
  assessmentId: ModelAssessmentIdSchema,
  environmentId: AssessmentEnvironmentIdSchema,
  targetId: ModelOfferingTargetIdSchema,
  displayName: NonEmptyString,
  description: Schema.String,
  license: NonEmptyString,
  profile: ServingProfileSchema,
  downloadBytes: NonNegativeSafeInteger,
  quantization: NonEmptyString,
  quantizationName: NonEmptyString,
  memory: Schema.Array(MemoryAssessmentSchema),
  performance: GenerationPerformanceSamplesSchema,
  capabilities: ModelCapabilitiesSchema,
  recommendationEvidence: Schema.optionalWith(
    LocalModelCatalogRecommendationEvidenceSchema,
    { as: "Option", exact: true },
  ),
  sources: Schema.Array(Schema.Struct({
    source: ModelPackageSourceSchema,
    files: Schema.Array(Schema.Struct({
      path: NonEmptyString,
      sha256: Sha256Digest,
    })),
  })),
} as const
export const LocalModelCatalogCandidateMetadataSchema = Schema.Struct(
  LocalModelCatalogCandidateMetadataFields,
).pipe(Schema.filter((candidate) =>
  candidate.performance.at(-1)?.contextTokens === candidate.profile.contextLength,
{ message: () => "candidate performance must end at the serving profile context" }))
export type LocalModelCatalogCandidateMetadata =
  typeof LocalModelCatalogCandidateMetadataSchema.Type

export const LocalModelCatalogCandidateSchema = Schema.Struct({
  ...LocalModelCatalogCandidateMetadataFields,
  download: LocalModelDownloadSchema,
  availability: LocalModelCatalogCandidateAvailabilitySchema,
}).pipe(Schema.filter((candidate) =>
  candidate.performance.at(-1)?.contextTokens === candidate.profile.contextLength,
{ message: () => "candidate performance must end at the serving profile context" }))
export type LocalModelCatalogCandidate = typeof LocalModelCatalogCandidateSchema.Type

export const LocalModelRecommendationSchema = Schema.Struct({
  id: RecommendationIdSchema,
  intent: Schema.Literal("balanced", "best_quality", "fastest", "lightweight"),
  explanation: Schema.String,
  candidate: LocalModelCatalogCandidateSchema,
})
export type LocalModelRecommendation = typeof LocalModelRecommendationSchema.Type

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

export const LocalModelRecommendationsLifecycleSchema = Schema.Union(
  Schema.TaggedStruct("Loading", {
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Ready", {
    entries: Schema.Array(LocalModelRecommendationSchema),
    catalog: Schema.Array(LocalModelCatalogCandidateSchema),
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
  Schema.TaggedStruct("Failed", {
    failure: ModelFailureSchema,
    progress: Schema.Array(LocalModelRecommendationProgressStepSchema),
  }),
)
export type LocalModelRecommendationsLifecycle =
  typeof LocalModelRecommendationsLifecycleSchema.Type

export const LocalModelsStateSchema = Schema.Struct({
  models: Schema.Array(LocalModelSchema),
  recommendations: LocalModelRecommendationsLifecycleSchema,
})
export type LocalModelsState = typeof LocalModelsStateSchema.Type

export const ModelPackagesStateSchema = Schema.Struct({
  entries: Schema.Array(ModelPackageEntrySchema),
})
export type ModelPackagesState = typeof ModelPackagesStateSchema.Type

export const ModelRecommendationsStateSchema = Schema.Struct({
  recommendations: Schema.Array(RecommendationSchema),
  failure: Schema.optionalWith(ModelFailureSchema, { as: "Option", exact: true }),
})
export type ModelRecommendationsState = typeof ModelRecommendationsStateSchema.Type

export const LocalModelDownloadsStateSchema = Schema.Struct({
  attempts: Schema.Array(DownloadAttemptSchema),
})
export type LocalModelDownloadsState = typeof LocalModelDownloadsStateSchema.Type

export const LocalProviderOfferingSchema = Schema.Struct({
  providerModelId: ProviderModelIdSchema,
  targetId: ModelOfferingTargetIdSchema,
  configuration: ModelServingConfigurationSchema,
  capabilities: ModelCapabilitiesSchema,
})
export type LocalProviderOffering = typeof LocalProviderOfferingSchema.Type

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

export const SlotSelectionSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  reasoningEffort: ReasoningEffortSchema,
})
export type SlotSelection = typeof SlotSelectionSchema.Type

export class ModelSlotUnassigned extends Schema.TaggedClass<ModelSlotUnassigned>()("Unassigned", {
  slotId: SlotIdSchema,
}) {}

export const ModelSlotDescriptorSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  displayName: NonEmptyString,
})
export type ModelSlotDescriptor = typeof ModelSlotDescriptorSchema.Type

export const ModelSlotAvailabilitySchema = Schema.Union(
  Schema.TaggedStruct("Available", {}),
  Schema.TaggedStruct("Unavailable", { failure: ModelFailureSchema }),
)
export type ModelSlotAvailability = typeof ModelSlotAvailabilitySchema.Type

export const ModelReleaseReasonSchema = Schema.Literal(
  "user_stop",
  "idle_timeout",
  "replacement",
  "memory_pressure",
)
export type ModelReleaseReason = typeof ModelReleaseReasonSchema.Type

export const ModelSlotInstanceLifecycleSchema = Schema.Union(
  Schema.TaggedStruct("Loading", {
    stage: Schema.Literal("queued", "resolving", "unloading", "loading", "verifying"),
    progress: Schema.optionalWith(Schema.Number.pipe(Schema.finite(), Schema.between(0, 1)), {
      as: "Option",
      exact: true,
    }),
    plannedAllocation: Schema.optionalWith(ModelLoadPlanSchema, { as: "Option", exact: true }),
  }),
  Schema.TaggedStruct("Ready", { allocation: ModelInstanceAllocationSchema }),
  Schema.TaggedStruct("Stopping", {
    reason: ModelReleaseReasonSchema,
    allocation: ModelStoppingAllocationSchema,
  }),
  Schema.TaggedStruct("Stopped", {
    reason: ModelReleaseReasonSchema,
  }),
  Schema.TaggedStruct("Failed", { failure: ModelFailureSchema }),
)
export type ModelSlotInstanceLifecycle = typeof ModelSlotInstanceLifecycleSchema.Type

export const ModelSlotInstanceSchema = Schema.Struct({
  id: ModelInstanceIdSchema,
  configurationId: ModelServingConfigurationIdSchema,
  lifecycle: ModelSlotInstanceLifecycleSchema,
})
export type ModelSlotInstance = typeof ModelSlotInstanceSchema.Type

export const ModelLoadResultSchema = Schema.Union(
  Schema.TaggedStruct("Ready", {
    instanceId: ModelInstanceIdSchema,
    configurationId: ModelServingConfigurationIdSchema,
  }),
  Schema.TaggedStruct("Cancelled", {
    instanceId: ModelInstanceIdSchema,
    reason: ModelReleaseReasonSchema,
  }),
)
export type ModelLoadResult = typeof ModelLoadResultSchema.Type

export const ModelLoadAdmissionSchema = Schema.Struct({
  instanceId: ModelInstanceIdSchema,
})
export type ModelLoadAdmission = typeof ModelLoadAdmissionSchema.Type

export const ModelSlotActionSchema = Schema.Literal("Load", "Stop", "RetryLoad")
export type ModelSlotAction = typeof ModelSlotActionSchema.Type

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
  instance: Schema.optionalWith(ModelSlotInstanceSchema, { as: "Option", exact: true }),
  actions: Schema.Array(ModelSlotActionSchema),
}) {}

export const ModelSlotLifecycle = defineFSM(
  {
    Unassigned: ModelSlotUnassigned,
    ConfiguredRemote: ModelSlotConfiguredRemote,
    ConfiguredLocal: ModelSlotConfiguredLocal,
  },
  {
    Unassigned: ["ConfiguredRemote", "ConfiguredLocal"],
    ConfiguredRemote: ["Unassigned", "ConfiguredLocal"],
    ConfiguredLocal: ["Unassigned", "ConfiguredRemote"],
  } as const,
)

export const ModelSlotSchema = Schema.Union(
  ModelSlotUnassigned,
  ModelSlotConfiguredRemote,
  ModelSlotConfiguredLocal,
).pipe(Schema.filter((slot) => {
  if (slot._tag === "Unassigned") return true
  if (slot.selection.providerId !== slot.descriptor.providerId
    || slot.selection.providerModelId !== slot.descriptor.providerModelId) return false
  if (slot._tag === "ConfiguredRemote") {
    return slot.selection.providerId !== "local" && slot.actions.length === 0
  }
  if (slot.selection.providerId !== "local") return false
  const expectedActions: readonly ModelSlotAction[] = slot.availability._tag === "Available"
    ? Option.match(slot.instance, {
        onNone: () => ["Load"],
        onSome: (instance) => {
          switch (instance.lifecycle._tag) {
            case "Loading":
            case "Ready":
              return ["Stop"]
            case "Failed":
              return instance.lifecycle.failure.retryable ? ["RetryLoad"] : []
            case "Stopping":
              return []
            case "Stopped":
              return ["Load"]
          }
        },
      })
    : []
  return expectedActions.length === slot.actions.length
    && expectedActions.every((action, index) => action === slot.actions[index])
}, { message: () => "slot identity, provider kind, and actions must agree with canonical slot state" }))
export type ModelSlot = typeof ModelSlotSchema.Type

export const ModelSlotsStateSchema = Schema.Struct({
  slots: Schema.Struct({
    primary: ModelSlotSchema,
    secondary: ModelSlotSchema,
  }),
  recentModelIds: Schema.Struct({
    primary: Schema.Array(ProviderModelIdSchema),
    secondary: Schema.Array(ProviderModelIdSchema),
  }),
  favoriteModels: Schema.Array(ProviderModelIdentitySchema),
}).pipe(
  Schema.filter((state) => state.slots.primary.slotId === PRIMARY_SLOT_ID
    && state.slots.secondary.slotId === SECONDARY_SLOT_ID,
  { message: () => "each model slot state must carry its containing slot identity" }),
  Schema.filter((state) => {
    const local = [state.slots.primary, state.slots.secondary].filter(
      (slot): slot is ModelSlotConfiguredLocal => slot._tag === "ConfiguredLocal",
    )
    const instanceIdsByConfiguration = new Map<string, Set<ModelInstanceId>>()
    for (const slot of local) {
      if (Option.isNone(slot.instance)) continue
      const instance = slot.instance.value
      const ids = instanceIdsByConfiguration.get(instance.configurationId) ?? new Set<ModelInstanceId>()
      ids.add(instance.id)
      instanceIdsByConfiguration.set(instance.configurationId, ids)
    }
    return [...instanceIdsByConfiguration.values()].every((ids) => ids.size === 1)
  }, {
    message: () => "matching local slots must share one canonical model instance",
  }),
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
  warningReserveBytes: NonNegativeSafeInteger,
  assessReserveBytes: NonNegativeSafeInteger,
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
