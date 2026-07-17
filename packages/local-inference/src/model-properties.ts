import { Schema } from "effect"

export const LlamaCppTemplateReasoningControlSchema = Schema.Union(
  Schema.TaggedStruct("Omitted", {}),
  Schema.TaggedStruct("EnableThinkingKwarg", { enabled: Schema.Boolean }),
  Schema.TaggedStruct("ReasoningEffortKwarg", { value: Schema.String }),
  Schema.TaggedStruct("EnableThinkingAndReasoningEffortKwarg", {
    enabled: Schema.Boolean,
    value: Schema.String,
  }),
)
export type LlamaCppTemplateReasoningControl = typeof LlamaCppTemplateReasoningControlSchema.Type

export const LlamaCppReasoningOptionSchema = Schema.Struct({
  reasoningEffort: Schema.String.pipe(Schema.minLength(1), Schema.maxLength(256)),
  control: LlamaCppTemplateReasoningControlSchema,
})
export type LlamaCppReasoningOption = typeof LlamaCppReasoningOptionSchema.Type

const ReasoningInspectionFields = {
  probeProtocolVersion: Schema.String.pipe(Schema.minLength(1)),
  options: Schema.Array(LlamaCppReasoningOptionSchema).pipe(Schema.minItems(1)),
} as const

/** Final semantic output of the differential template probe. */
export const LlamaCppReasoningTemplateInspectionSchema = Schema.Struct(ReasoningInspectionFields)
export type LlamaCppReasoningTemplateInspection = typeof LlamaCppReasoningTemplateInspectionSchema.Type

/** Cached result for one exact serving route configuration. */
export const LlamaCppReasoningInspectionSchema = Schema.Struct({
  routeId: Schema.String.pipe(Schema.minLength(1)),
  fingerprint: Schema.String.pipe(Schema.minLength(1)),
  ...ReasoningInspectionFields,
})
export type LlamaCppReasoningInspection = typeof LlamaCppReasoningInspectionSchema.Type

/** Cached vision result from a loaded child's `/props`. */
export const LlamaCppVisionInspectionSchema = Schema.Struct({
  routeId: Schema.String.pipe(Schema.minLength(1)),
  fingerprint: Schema.String.pipe(Schema.minLength(1)),
  value: Schema.Boolean,
})
export type LlamaCppVisionInspection = typeof LlamaCppVisionInspectionSchema.Type

/** Path-owned cache containing only concrete route inspection results. */
export const LocalModelDiscoveredPropertiesSchema = Schema.Struct({
  modelPath: Schema.String.pipe(Schema.minLength(1)),
  visionInspections: Schema.Array(LlamaCppVisionInspectionSchema),
  reasoningInspections: Schema.Array(LlamaCppReasoningInspectionSchema),
})
export type LocalModelDiscoveredProperties = typeof LocalModelDiscoveredPropertiesSchema.Type
