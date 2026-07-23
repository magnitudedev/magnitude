import { Schema } from "effect"
import { ReasoningEffortSchema } from "@magnitudedev/ai"
import {
  ModelArtifactFingerprintSchema,
  ModelRecipeCatalogModelIdSchema,
  ModelRecipeConfigurationIdSchema,
  NativeIcnModelIdSchema,
} from "../provider/model-identity.js"

const NonNegativeNumber = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
const PositiveNumber = Schema.Number.pipe(Schema.finite(), Schema.positive())
const PositiveInteger = Schema.Number.pipe(Schema.int(), Schema.positive())

export const ModelRecipeFitClass = Schema.Literal(
  "full_accelerator",
  "hybrid",
  "cpu_or_unified",
  "unknown",
)
export type ModelRecipeFitClass = Schema.Schema.Type<typeof ModelRecipeFitClass>

export const ModelRecipeQuantization = Schema.Struct({
  format: Schema.String,
  quantAwareCheckpoint: Schema.Boolean,
  fidelityLabel: Schema.String,
  fidelityEvidence: Schema.String,
  fidelitySourceUrl: Schema.String,
})
export type ModelRecipeQuantization = Schema.Schema.Type<typeof ModelRecipeQuantization>

export const ModelRecipeRecommendationIntent = Schema.Literal(
  "balanced",
  "best_quality",
  "fastest",
  "lightweight",
)
export type ModelRecipeRecommendationIntent = Schema.Schema.Type<typeof ModelRecipeRecommendationIntent>

export const ModelRecipeGenerationEstimate = Schema.Struct({
  contextTokens: PositiveInteger,
  lowerTokensPerSecond: PositiveNumber,
  expectedTokensPerSecond: PositiveNumber,
  upperTokensPerSecond: PositiveNumber,
  confidence: Schema.Literal("high", "moderate", "low"),
  method: Schema.String,
})
export type ModelRecipeGenerationEstimate = Schema.Schema.Type<typeof ModelRecipeGenerationEstimate>

export const ModelRecipeRecommendation = Schema.Struct({
  configurationId: ModelRecipeConfigurationIdSchema,
  catalogModelId: ModelRecipeCatalogModelIdSchema,
  artifactFingerprint: ModelArtifactFingerprintSchema,
  modelId: Schema.optionalWith(NativeIcnModelIdSchema, { as: "Option", exact: true }),
  intent: ModelRecipeRecommendationIntent,
  displayName: Schema.String,
  family: Schema.String,
  architecture: Schema.Literal("dense", "moe"),
  capabilities: Schema.Struct({
    vision: Schema.Boolean,
    tools: Schema.Boolean,
    structuredOutput: Schema.Boolean,
    reasoningEfforts: Schema.Array(ReasoningEffortSchema),
    defaultReasoningEffort: Schema.optionalWith(ReasoningEffortSchema, { as: "Option", exact: true }),
  }),
  totalParametersBillions: Schema.optionalWith(NonNegativeNumber, { as: "Option", exact: true }),
  activeParametersBillions: Schema.optionalWith(NonNegativeNumber, { as: "Option", exact: true }),
  effectiveParametersBillions: Schema.optionalWith(NonNegativeNumber, { as: "Option", exact: true }),
  quantization: ModelRecipeQuantization,
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
  contextWindow: PositiveInteger,
  estimatedRuntimeBytes: NonNegativeNumber,
  stableCapacityBudgetBytes: NonNegativeNumber,
  fitMarginBytes: Schema.Number.pipe(Schema.finite()),
  fitClass: ModelRecipeFitClass,
  constrainedContext: Schema.Boolean,
  estimatedGeneration: Schema.optionalWith(ModelRecipeGenerationEstimate, { as: "Option", exact: true }),
  explanation: Schema.String,
})
export type ModelRecipeRecommendation = Schema.Schema.Type<typeof ModelRecipeRecommendation>

export const ModelRecipesState = Schema.Union(
  Schema.TaggedStruct("Loading", {}),
  Schema.TaggedStruct("Ready", {
    recommendations: Schema.Array(ModelRecipeRecommendation),
    failureCount: Schema.Number.pipe(Schema.int(), Schema.nonNegative()),
  }),
  Schema.TaggedStruct("Failed", { message: Schema.String }),
)
export type ModelRecipesState = Schema.Schema.Type<typeof ModelRecipesState>
