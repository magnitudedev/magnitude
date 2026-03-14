import { Schema } from 'effect'

const NullableOptional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(Schema.NullishOr(schema), {
    default: () => null as A | null,
  })

export const ModelSelectionSchema = Schema.Struct({
  providerId: Schema.String,
  modelId: Schema.String,
})
export type ModelSelection = Schema.Schema.Type<typeof ModelSelectionSchema>

export const ProviderOptionsSchema = Schema.Struct({
  baseUrl: Schema.optional(Schema.String),
  region: Schema.optional(Schema.String),
  project: Schema.optional(Schema.String),
  location: Schema.optional(Schema.String),
  modelId: Schema.optional(Schema.String),
}).pipe(Schema.extend(Schema.Record({ key: Schema.String, value: Schema.Unknown })))
export type ProviderOptions = Schema.Schema.Type<typeof ProviderOptionsSchema>

export const ContextLimitPolicySchema = Schema.Struct({
  softCapRatio: Schema.optional(Schema.Number),
  softCapMaxTokens: NullableOptional(Schema.Number),
})
export interface ContextLimitPolicy extends Omit<Schema.Schema.Type<typeof ContextLimitPolicySchema>, 'softCapMaxTokens'> {
  softCapMaxTokens: number | null
}

export const MagnitudeConfigSchema = Schema.Struct({
  primaryModel: NullableOptional(ModelSelectionSchema),
  secondaryModel: NullableOptional(ModelSelectionSchema),
  browserModel: NullableOptional(ModelSelectionSchema),
  providerOptions: Schema.optional(
    Schema.Record({ key: Schema.String, value: ProviderOptionsSchema })
  ),
  setupComplete: Schema.optional(Schema.Boolean),
  machineId: Schema.optional(Schema.String),
  telemetry: Schema.optional(Schema.Boolean),
  memory: Schema.optional(Schema.Boolean),
  contextLimits: Schema.optional(ContextLimitPolicySchema),
})
export interface MagnitudeConfig extends Omit<Schema.Schema.Type<typeof MagnitudeConfigSchema>, 'primaryModel' | 'secondaryModel' | 'browserModel'> {
  primaryModel: ModelSelection | null
  secondaryModel: ModelSelection | null
  browserModel: ModelSelection | null
}