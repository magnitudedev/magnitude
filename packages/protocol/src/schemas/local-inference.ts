import { Schema } from "effect"

const NonNegativeNumber = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const PositiveInteger = Schema.Number.pipe(Schema.int(), Schema.positive())
export const LocalModelRole = Schema.Literal("main", "subagent")
export type LocalModelRole = Schema.Schema.Type<typeof LocalModelRole>

export const LocalSessionConcurrency = Schema.Literal("one", "up_to_three")
export type LocalSessionConcurrency = Schema.Schema.Type<typeof LocalSessionConcurrency>

export const LocalInferenceUsageSelection = Schema.Struct({
  localModelRole: LocalModelRole,
  sessionConcurrency: LocalSessionConcurrency,
})
export type LocalInferenceUsageSelection = Schema.Schema.Type<typeof LocalInferenceUsageSelection>

export const LocalInferenceServingProfile = Schema.Struct({
  localModelRole: LocalModelRole,
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

export const LocalInferenceDistributionState = Schema.Union(
  Schema.TaggedStruct("Missing", {}),
  Schema.TaggedStruct("Unsupported", { message: Schema.String }),
  Schema.TaggedStruct("Invalid", { message: Schema.String }),
  Schema.TaggedStruct("Ready", {
    build: PositiveInteger,
    source: Schema.Literal("managed", "configured"),
  }),
)
export type LocalInferenceDistributionState = Schema.Schema.Type<typeof LocalInferenceDistributionState>

export const LocalInferenceHostProfile = Schema.Struct({
  systemMemoryBytes: NonNegativeNumber,
  cpuModel: Schema.NullOr(Schema.String),
  logicalCores: PositiveInteger,
  memoryDomains: Schema.Array(Schema.Struct({
    id: Schema.String,
    kind: Schema.Literal("system", "physical_device", "unified_working_set"),
    stableCapacityBytes: NonNegativeNumber,
    currentFreeBytes: Schema.NullOr(NonNegativeNumber),
    sharesSystemMemory: Schema.Boolean,
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
    sizeBytes: NonNegativeNumber,
    sha256: Schema.String,
    downloadUrl: Schema.String,
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

const ChoiceFields = {
  choiceId: Schema.String,
  displayName: Schema.String,
  providerModelId: Schema.String,
  contextTokens: Schema.optional(PositiveInteger),
  fitClass: LocalInferenceFitClass,
  compatible: Schema.Boolean,
  explanation: Schema.String,
  residency: Schema.Literal("loaded", "sleeping", "unloaded", "loading", "failed"),
  quantization: Schema.optional(LocalInferenceQuantization),
  sizeBytes: Schema.optional(NonNegativeNumber),
  servingProfile: Schema.optional(LocalInferenceServingProfile),
}

export const LocalModelChoice = Schema.Union(
  Schema.TaggedStruct("RunningExternal", ChoiceFields),
  Schema.TaggedStruct("RunningManaged", ChoiceFields),
  Schema.TaggedStruct("StoredOwned", ChoiceFields),
  Schema.TaggedStruct("StoredExternal", ChoiceFields),
)
export type LocalModelChoice = Schema.Schema.Type<typeof LocalModelChoice>

export const ActiveLocalBindingSummary = Schema.Union(
  Schema.TaggedStruct("Managed", {
    selectionId: Schema.String,
    providerModelId: Schema.String,
    contextTokens: PositiveInteger,
  }),
  Schema.TaggedStruct("External", {
    selectionId: Schema.String,
    providerModelId: Schema.String,
    contextTokens: PositiveInteger,
  }),
)
export type ActiveLocalBindingSummary = Schema.Schema.Type<typeof ActiveLocalBindingSummary>

export const LocalInferenceOperationStage = Schema.Literal(
  "queued", "resolving_files", "writing_preset", "starting_router",
  "unloading_previous", "loading", "verifying", "loaded",
)
export type LocalInferenceOperationStage = Schema.Schema.Type<typeof LocalInferenceOperationStage>

export const LocalInferenceOperationSnapshot = Schema.Struct({
  operationId: Schema.String,
  providerModelId: Schema.String,
  status: Schema.Literal("running", "completed", "failed"),
  stage: LocalInferenceOperationStage,
  progress: Schema.optional(Schema.Number.pipe(Schema.finite(), Schema.between(0, 1))),
  message: Schema.optional(Schema.String),
})
export type LocalInferenceOperationSnapshot = Schema.Schema.Type<typeof LocalInferenceOperationSnapshot>

export const LocalInferenceErrorCode = Schema.Literal(
  "distribution_missing",
  "unsupported_platform",
  "invalid_selection",
  "artifact_unavailable",
  "license_required",
  "insufficient_disk_space",
  "integrity_failed",
  "artifact_not_owned",
  "artifact_active",
  "context_mismatch",
  "server_start_failed",
  "external_server_unavailable",
  "configuration_failed",
  "runtime_probe_failed",
  "cancelled",
)
export type LocalInferenceErrorCode = Schema.Schema.Type<typeof LocalInferenceErrorCode>

export const LocalInferenceState = Schema.Struct({
  usage: Schema.NullOr(LocalInferenceUsageSelection),
  activeBinding: Schema.NullOr(ActiveLocalBindingSummary),
  distribution: LocalInferenceDistributionState,
  host: LocalInferenceHostState,
  choices: Schema.Array(LocalModelChoice),
  operations: Schema.Array(LocalInferenceOperationSnapshot),
  recommendations: Schema.Array(LocalModelRecommendation),
  warnings: Schema.Array(LocalInferenceWarning),
})
export type LocalInferenceState = Schema.Schema.Type<typeof LocalInferenceState>
