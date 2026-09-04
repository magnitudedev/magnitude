import { Schema } from "effect"

/** Public wire values; no daemon, host UI, or private model lifecycle dependencies. */
export const IntegrationModelIdSchema = Schema.NonEmptyString.pipe(Schema.brand("IntegrationModelId"))
export const JsonInstallationStateSchema = Schema.Literal("not_installed", "installing", "installed", "removing", "unavailable")
export const JsonResidencyStateSchema = Schema.Literal("unloaded", "loading", "ready", "stopping", "failed")
export const JsonLocalModelSchema = Schema.Struct({
  modelId: IntegrationModelIdSchema,
  displayName: Schema.NonEmptyString,
  installation: JsonInstallationStateSchema,
  residency: Schema.optionalWith(JsonResidencyStateSchema, { as: "Option", exact: true }),
})
export type JsonLocalModel = typeof JsonLocalModelSchema.Type
export const ModelsStatusJsonDataSchema = Schema.Union(
  Schema.Struct({ state: Schema.Literal("initializing"), models: Schema.Tuple() }),
  Schema.Struct({ state: Schema.Literal("ready"), models: Schema.Array(JsonLocalModelSchema) }),
)
export type ModelsStatusJsonData = typeof ModelsStatusJsonDataSchema.Type
export const ModelsLoadJsonDataSchema = Schema.Struct({ modelId: IntegrationModelIdSchema })
export const ModelsStopJsonDataSchema = Schema.Struct({})
export const JsonCommandNameSchema = Schema.Literal("models.status", "models.load", "models.stop")
export type JsonCommandName = typeof JsonCommandNameSchema.Type

export const jsonSuccessEnvelopeSchema = <C extends string, A, I>(command: C, data: Schema.Schema<A, I>) => Schema.Struct({
  schemaVersion: Schema.Literal(1), command: Schema.Literal(command), ok: Schema.Literal(true), data,
})
export const jsonFailureEnvelopeSchema = <C extends string>(command: C) => Schema.Struct({
  schemaVersion: Schema.Literal(1), command: Schema.Literal(command), ok: Schema.Literal(false),
  error: Schema.Struct({ message: Schema.String }),
})
export const ModelsStatusEnvelopeSchema = jsonSuccessEnvelopeSchema("models.status", ModelsStatusJsonDataSchema)
export const ModelsLoadEnvelopeSchema = jsonSuccessEnvelopeSchema("models.load", ModelsLoadJsonDataSchema)
export const ModelsStopEnvelopeSchema = jsonSuccessEnvelopeSchema("models.stop", ModelsStopJsonDataSchema)

const Count = Schema.Number.pipe(Schema.int(), Schema.nonNegative())
const Duration = Schema.Number.pipe(Schema.finite(), Schema.nonNegative())
export const MagnitudeProgressSchema = Schema.Union(
  Schema.Struct({ phase: Schema.Literal("model_loading"), fraction: Schema.Number.pipe(Schema.finite()) }),
  Schema.Struct({ phase: Schema.Literal("queued") }),
  Schema.Struct({ phase: Schema.Literal("preparing") }),
  Schema.Struct({ phase: Schema.Literal("prefill"), completed_tokens: Count, total_tokens: Count, cached_tokens: Count }),
  Schema.Struct({ phase: Schema.Literal("generating") }),
)
export type MagnitudeProgress = typeof MagnitudeProgressSchema.Type
/** Timings are cumulative snapshots within one request, never additive deltas. */
export const MagnitudeTimingsSchema = Schema.Struct({
  prompt_ms: Duration, time_to_first_token_ms: Duration,
  predicted_n: Count, predicted_ms: Duration, predicted_per_second: Duration,
})
export type MagnitudeTimings = typeof MagnitudeTimingsSchema.Type
