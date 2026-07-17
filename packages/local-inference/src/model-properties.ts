import { Schema } from "effect"
import { LlamaCppReasoningProfileSchema } from "./llamacpp/reasoning-profile"
export * from "./llamacpp/reasoning-profile"

const ReasoningTemplateInspectionFields = {
  profile: LlamaCppReasoningProfileSchema,
} as const

/** Final resolved reasoning profile produced by inspecting one loaded template. */
export const LlamaCppReasoningTemplateInspectionSchema = Schema.Struct(ReasoningTemplateInspectionFields)
export type LlamaCppReasoningTemplateInspection = typeof LlamaCppReasoningTemplateInspectionSchema.Type

/** Cached reasoning profile for one exact serving route configuration. */
export const LlamaCppReasoningInspectionSchema = Schema.Struct({
  routeId: Schema.String.pipe(Schema.minLength(1)),
  fingerprint: Schema.String.pipe(Schema.minLength(1)),
  ...ReasoningTemplateInspectionFields,
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
