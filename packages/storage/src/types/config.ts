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

export const RoleConfigSchema = Schema.Struct({
  model: Schema.NullishOr(ModelSelectionSchema),
})
export type RoleConfig = Schema.Schema.Type<typeof RoleConfigSchema>

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
  roles: Schema.optional(
    Schema.Record({ key: Schema.String, value: RoleConfigSchema })
  ),
  // used for provider options and local provider options like base URLs, API key etc., not used for oauth atm
  providers: Schema.optional(
    Schema.Record({ key: Schema.String, value: ProviderOptionsSchema })
  ),
  setupComplete: Schema.optional(Schema.Boolean),
  machineId: Schema.optional(Schema.String),
  telemetry: Schema.optional(Schema.Boolean),
  contextLimits: Schema.optional(ContextLimitPolicySchema),
})
export interface MagnitudeConfig extends Omit<Schema.Schema.Type<typeof MagnitudeConfigSchema>, 'roles'> {
  roles: Record<string, RoleConfig>
}
