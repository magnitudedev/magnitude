import { Schema } from "effect"
import { ProviderModelAvailabilitySchema, ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"
import { MirroredSnapshotSchema } from "./mirrored-resource"

const NonNegativeNumber = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const PositiveInteger = Schema.Number.pipe(Schema.int(), Schema.positive())
export const LocalSessionConcurrency = Schema.Literal("one", "up_to_three")
export type LocalSessionConcurrency = Schema.Schema.Type<typeof LocalSessionConcurrency>

export const LocalInferenceUsageSelection = Schema.Struct({
  sessionConcurrency: LocalSessionConcurrency,
})
export type LocalInferenceUsageSelection = Schema.Schema.Type<typeof LocalInferenceUsageSelection>

export const LocalInferenceServingProfile = Schema.Struct({
  sessionConcurrency: LocalSessionConcurrency,
  parallelSlots: PositiveInteger,
  contextTokensPerSlot: PositiveInteger,
  totalContextCapacityTokens: PositiveInteger,
  slotAllocation: Schema.Literal("uniform"),
  runtimeProfileId: Schema.String,
})
export type LocalInferenceServingProfile = Schema.Schema.Type<typeof LocalInferenceServingProfile>

export const LocalInferenceFitClass = Schema.Literal("full_accelerator", "hybrid", "cpu_or_unified", "unknown")
export type LocalInferenceFitClass = Schema.Schema.Type<typeof LocalInferenceFitClass>

export const LocalModelFitAssessmentSchema = Schema.Union(
  Schema.TaggedStruct("NotAssessed", {}),
  Schema.TaggedStruct("Assessed", {
    requiredTotalBytes: NonNegativeNumber,
    domains: Schema.Array(Schema.Struct({
      memoryDomainId: Schema.String.pipe(Schema.minLength(1)),
      requiredBytes: NonNegativeNumber,
      stableCapacityBytes: NonNegativeNumber,
      marginBytes: Schema.Number.pipe(Schema.finite()),
    })).pipe(Schema.minItems(1)),
    result: Schema.Literal("fits", "does_not_fit"),
  }),
)
export type LocalModelFitAssessment = Schema.Schema.Type<typeof LocalModelFitAssessmentSchema>

export const LocalInferenceQuantization = Schema.Struct({
  format: Schema.String,
  quantAwareCheckpoint: Schema.Boolean,
  fidelityLabel: Schema.String,
  fidelityEvidence: Schema.String,
  fidelitySourceUrl: Schema.String,
})
export type LocalInferenceQuantization = Schema.Schema.Type<typeof LocalInferenceQuantization>

export const LocalInferenceWarning = Schema.Struct({
  code: Schema.String,
  message: Schema.String,
})
export type LocalInferenceWarning = Schema.Schema.Type<typeof LocalInferenceWarning>

export const LocalInferenceHostProfile = Schema.Struct({
  platform: Schema.String,
  architecture: Schema.String,
  systemMemoryBytes: NonNegativeNumber,
  cpuModel: Schema.NullOr(Schema.String),
  logicalCores: PositiveInteger,
  memoryDomains: Schema.Array(Schema.Struct({
    id: Schema.String,
    kind: Schema.Literal("system", "physical_device", "unified_memory"),
    totalCapacityBytes: NonNegativeNumber,
    stableCapacityBytes: NonNegativeNumber,
    currentFreeBytes: Schema.NullOr(NonNegativeNumber),
    sharesSystemMemory: Schema.Boolean,
    backendNames: Schema.Array(Schema.String),
    deviceNames: Schema.Array(Schema.String),
    splitGroupId: Schema.NullOr(Schema.String),
  })),
})
export type LocalInferenceHostProfile = Schema.Schema.Type<typeof LocalInferenceHostProfile>

export const LocalInferenceHostState = Schema.Union(
  Schema.TaggedStruct("Available", { profile: LocalInferenceHostProfile }),
  Schema.TaggedStruct("Unavailable", { message: Schema.String }),
)
export type LocalInferenceHostState = Schema.Schema.Type<typeof LocalInferenceHostState>

export const LocalModelRecommendation = Schema.Struct({
  configurationId: Schema.String,
  catalogModelId: Schema.String,
  badge: Schema.Literal("recommended", "lighter", "higher_fidelity", "alternative"),
  displayName: Schema.String,
  family: Schema.String,
  architecture: Schema.Literal("dense", "moe"),
  totalParametersBillions: Schema.optional(NonNegativeNumber),
  activeParametersBillions: Schema.optional(NonNegativeNumber),
  effectiveParametersBillions: Schema.optional(NonNegativeNumber),
  quantization: LocalInferenceQuantization,
  quantTag: Schema.String,
  repo: Schema.String,
  revision: Schema.String,
  files: Schema.Array(Schema.Struct({
    path: Schema.String,
    role: Schema.Literal("weights", "shard", "projector", "auxiliary", "draft", "mtp"),
    sizeBytes: NonNegativeNumber,
    sha256: Schema.String,
  })),
  totalDownloadBytes: NonNegativeNumber,
  sourcePageUrl: Schema.String,
  license: Schema.Struct({
    id: Schema.String,
    url: Schema.String,
    acknowledgementRequired: Schema.Boolean,
  }),
  contextTokens: PositiveInteger,
  servingProfile: LocalInferenceServingProfile,
  modelMaximumContextTokens: PositiveInteger,
  estimatedRuntimeBytes: NonNegativeNumber,
  stableCapacityBudgetBytes: NonNegativeNumber,
  fitMarginBytes: Schema.Number.pipe(Schema.finite()),
  fitClass: LocalInferenceFitClass,
  constrainedContext: Schema.Boolean,
  explanation: Schema.String,
})
export type LocalModelRecommendation = Schema.Schema.Type<typeof LocalModelRecommendation>

export const LocalInferenceRecommendationState = Schema.Union(
  Schema.TaggedStruct("NotRequested", {}),
  Schema.TaggedStruct("Loading", {}),
  Schema.TaggedStruct("Ready", {
    recommendations: Schema.Array(LocalModelRecommendation),
  }),
  Schema.TaggedStruct("Failed", {
    message: Schema.String,
  }),
)
export type LocalInferenceRecommendationState = Schema.Schema.Type<typeof LocalInferenceRecommendationState>

const ChoiceFields = {
  choiceId: Schema.String,
  displayName: Schema.String,
  providerModelId: ProviderModelIdSchema,
  contextTokens: Schema.optional(PositiveInteger),
  fitClass: LocalInferenceFitClass,
  availability: ProviderModelAvailabilitySchema,
  fitAssessment: LocalModelFitAssessmentSchema,
  explanation: Schema.String,
  residency: Schema.Literal("loaded", "sleeping", "unloaded", "loading", "failed"),
  quantization: Schema.optional(LocalInferenceQuantization),
  sizeBytes: Schema.optional(NonNegativeNumber),
  servingProfile: Schema.optional(LocalInferenceServingProfile),
}

export const LocalModelChoice = Schema.Union(
  Schema.TaggedStruct("Running", ChoiceFields),
  Schema.TaggedStruct("Stored", ChoiceFields),
)
export type LocalModelChoice = Schema.Schema.Type<typeof LocalModelChoice>

export const ActiveLocalBindingSummary = Schema.Struct({
  selectionId: Schema.String,
  providerModelId: ProviderModelIdSchema,
  contextTokens: PositiveInteger,
})
export type ActiveLocalBindingSummary = Schema.Schema.Type<typeof ActiveLocalBindingSummary>

export const LocalInferenceOperationStage = Schema.Literal(
  "queued", "resolving", "checking_space", "downloading", "publishing",
  "assessing", "unloading", "loading", "verifying", "ready",
)
export type LocalInferenceOperationStage = Schema.Schema.Type<typeof LocalInferenceOperationStage>

export const LocalInferenceOperationSnapshot = Schema.Struct({
  operationId: Schema.String,
  providerModelId: ProviderModelIdSchema,
  status: Schema.Literal("running", "completed", "failed"),
  stage: LocalInferenceOperationStage,
  progress: Schema.optional(Schema.Number.pipe(Schema.finite(), Schema.between(0, 1))),
  message: Schema.optional(Schema.String),
})
export type LocalInferenceOperationSnapshot = Schema.Schema.Type<typeof LocalInferenceOperationSnapshot>

export const LocalInferenceErrorCode = Schema.Literal(
  "icn_unavailable",
  "unsupported_platform",
  "invalid_selection",
  "artifact_unavailable",
  "license_required",
  "insufficient_disk_space",
  "integrity_failed",
  "artifact_active",
  "context_mismatch",
  "runtime_start_failed",
  "configuration_failed",
  "runtime_probe_failed",
  "cancelled",
)
export type LocalInferenceErrorCode = Schema.Schema.Type<typeof LocalInferenceErrorCode>

export const LocalInferenceState = Schema.Struct({
  usage: Schema.NullOr(LocalInferenceUsageSelection),
  activeBinding: Schema.NullOr(ActiveLocalBindingSummary),
  host: LocalInferenceHostState,
  choices: Schema.Array(LocalModelChoice),
  operations: Schema.Array(LocalInferenceOperationSnapshot),
  recommendationState: LocalInferenceRecommendationState,
  warnings: Schema.Array(LocalInferenceWarning),
})
export type LocalInferenceState = Schema.Schema.Type<typeof LocalInferenceState>

export const LocalInferenceSnapshotSchema = MirroredSnapshotSchema(LocalInferenceState)
export type LocalInferenceSnapshot = Schema.Schema.Type<typeof LocalInferenceSnapshotSchema>
