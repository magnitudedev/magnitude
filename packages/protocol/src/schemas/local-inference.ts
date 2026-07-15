import { Schema } from "effect"
import { StreamHeartbeat } from "./events"

const NonNegativeNumber = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const PositiveInteger = Schema.Number.pipe(Schema.int(), Schema.positive())

export const LocalInferenceCapacityKind = Schema.Literal(
  "physical-device-memory",
  "recommended-working-set",
  "none",
  "unknown",
)
export type LocalInferenceCapacityKind = Schema.Schema.Type<typeof LocalInferenceCapacityKind>

export const LocalInferenceAccelerator = Schema.Struct({
  id: Schema.String,
  backend: Schema.String,
  description: Schema.String,
  capacityBytes: Schema.optional(NonNegativeNumber),
  capacityKind: LocalInferenceCapacityKind,
  memoryDomainId: Schema.String,
  /** Present only when the managed backend can split one model across the named device group. */
  modelSplitGroupId: Schema.optional(Schema.String),
  sharesSystemMemory: Schema.Union(Schema.Boolean, Schema.Literal("unknown")),
  /** Transient diagnostic only. Recommendation policy must never consume this field. */
  currentFreeBytes: Schema.optional(NonNegativeNumber),
})
export type LocalInferenceAccelerator = Schema.Schema.Type<typeof LocalInferenceAccelerator>

export const LocalInferenceWarning = Schema.Struct({
  code: Schema.String,
  message: Schema.String,
})
export type LocalInferenceWarning = Schema.Schema.Type<typeof LocalInferenceWarning>

export const LocalInferenceCapabilities = Schema.Struct({
  binary: Schema.Struct({
    identity: Schema.String,
    version: Schema.optional(Schema.String),
  }),
  system: Schema.Struct({
    totalMemoryBytes: NonNegativeNumber,
    cpuModel: Schema.optional(Schema.String),
    logicalCores: Schema.optional(PositiveInteger),
  }),
  accelerators: Schema.Array(LocalInferenceAccelerator),
  warnings: Schema.Array(LocalInferenceWarning),
})
export type LocalInferenceCapabilities = Schema.Schema.Type<typeof LocalInferenceCapabilities>

export const LocalInferenceFitClass = Schema.Literal(
  "full_accelerator",
  "hybrid",
  "cpu_or_unified",
  "unknown",
)
export type LocalInferenceFitClass = Schema.Schema.Type<typeof LocalInferenceFitClass>

export const LocalInferenceQuantization = Schema.Struct({
  format: Schema.String,
  bitsClass: Schema.Literal("q4", "q5", "q6", "q8", "fp8", "mxfp4", "other"),
  quantAwareCheckpoint: Schema.Boolean,
  fidelityLabel: Schema.String,
  fidelityEvidence: Schema.String,
  fidelitySourceUrl: Schema.String,
})
export type LocalInferenceQuantization = Schema.Schema.Type<typeof LocalInferenceQuantization>

export const LocalInferenceArtifactFile = Schema.Struct({
  path: Schema.String,
  sizeBytes: NonNegativeNumber,
  sha256: Schema.String,
  downloadUrl: Schema.String,
})
export type LocalInferenceArtifactFile = Schema.Schema.Type<typeof LocalInferenceArtifactFile>

export const LocalModelRecommendation = Schema.Struct({
  configurationId: Schema.String,
  catalogModelId: Schema.String,
  badge: Schema.Literal("recommended", "lighter", "higher_fidelity"),
  displayName: Schema.String,
  family: Schema.String,
  architecture: Schema.Literal("dense", "moe"),
  totalParametersBillions: Schema.optional(NonNegativeNumber),
  activeParametersBillions: Schema.optional(NonNegativeNumber),
  quantization: LocalInferenceQuantization,
  repo: Schema.String,
  revision: Schema.String,
  quantTag: Schema.String,
  files: Schema.Array(LocalInferenceArtifactFile),
  totalDownloadBytes: NonNegativeNumber,
  sourcePageUrl: Schema.String,
  license: Schema.Struct({
    id: Schema.String,
    url: Schema.String,
    acknowledgementRequired: Schema.Boolean,
  }),
  contextTokens: PositiveInteger,
  modelMaximumContextTokens: PositiveInteger,
  estimatedRuntimeBytes: NonNegativeNumber,
  stableCapacityBudgetBytes: NonNegativeNumber,
  fitMarginBytes: NonNegativeNumber,
  fitClass: LocalInferenceFitClass,
  constrainedContext: Schema.Boolean,
  explanation: Schema.String,
})
export type LocalModelRecommendation = Schema.Schema.Type<typeof LocalModelRecommendation>

export const LocalModelChoice = Schema.Struct({
  choiceId: Schema.String,
  source: Schema.Literal("running", "downloaded"),
  displayName: Schema.String,
  providerModelId: Schema.String,
  serverId: Schema.optional(Schema.String),
  cacheId: Schema.optional(Schema.String),
  catalogModelId: Schema.optional(Schema.String),
  quantization: Schema.optional(LocalInferenceQuantization),
  sizeBytes: Schema.optional(NonNegativeNumber),
  totalParametersBillions: Schema.optional(NonNegativeNumber),
  activeParametersBillions: Schema.optional(NonNegativeNumber),
  contextTokens: PositiveInteger,
  modelMaximumContextTokens: Schema.optional(PositiveInteger),
  fitClass: LocalInferenceFitClass,
  managed: Schema.Boolean,
  compatible: Schema.Boolean,
  explanation: Schema.String,
})
export type LocalModelChoice = Schema.Schema.Type<typeof LocalModelChoice>

export const LocalInferenceOnboardingSnapshot = Schema.Struct({
  schemaVersion: PositiveInteger,
  onboarding: Schema.Struct({
    required: Schema.Boolean,
    completedVersion: Schema.optional(PositiveInteger),
    completedAt: Schema.optional(Schema.String),
  }),
  configuration: Schema.Struct({
    usable: Schema.Boolean,
  }),
  runtime: Schema.Struct({
    status: Schema.Literal("ready", "integration_pending", "error"),
    canDownload: Schema.Boolean,
    canActivate: Schema.Boolean,
    diagnostic: Schema.optional(Schema.String),
  }),
  capabilities: Schema.optional(LocalInferenceCapabilities),
  running: Schema.Array(LocalModelChoice),
  downloaded: Schema.Array(LocalModelChoice),
  recommendations: Schema.Array(LocalModelRecommendation),
  warnings: Schema.Array(LocalInferenceWarning),
})
export type LocalInferenceOnboardingSnapshot = Schema.Schema.Type<typeof LocalInferenceOnboardingSnapshot>

export const LocalModelDownloadProgress = Schema.Struct({
  operationId: Schema.String,
  status: Schema.Literal(
    "queued",
    "downloading",
    "verifying",
    "ready",
    "failed",
    "cancelled",
  ),
  completedBytes: NonNegativeNumber,
  totalBytes: NonNegativeNumber,
  currentFile: Schema.optional(Schema.String),
  bytesPerSecond: Schema.optional(NonNegativeNumber),
  resumable: Schema.Boolean,
  message: Schema.optional(Schema.String),
  selectionId: Schema.optional(Schema.String),
})
export type LocalModelDownloadProgress = Schema.Schema.Type<typeof LocalModelDownloadProgress>

export const LocalModelDownloadWireEvent = Schema.Union(LocalModelDownloadProgress, StreamHeartbeat)
export type LocalModelDownloadWireEvent = Schema.Schema.Type<typeof LocalModelDownloadWireEvent>
