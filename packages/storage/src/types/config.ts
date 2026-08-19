import { Schema } from 'effect'
import { CustomEndpointDeclarationsSchema } from './custom-endpoints'

const NullableOptional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(Schema.NullishOr(schema), {
    default: (): A | null => null,
  })

export const ContextLimitPolicySchema = Schema.Struct({
  softCapRatio: Schema.optional(Schema.Number),
  softCapMaxTokens: NullableOptional(Schema.Number),
})
export interface ContextLimitPolicy extends Omit<Schema.Schema.Type<typeof ContextLimitPolicySchema>, 'softCapMaxTokens'> {
  softCapMaxTokens: number | null
}

const SerializableOptional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: 'Option', exact: true } as const)

export const MagnitudeConfigSchema = Schema.Struct({
  contextLimits: Schema.optional(ContextLimitPolicySchema),
  providers: SerializableOptional(CustomEndpointDeclarationsSchema),
  checkForUpdateOnStartup: SerializableOptional(Schema.Boolean),
})

export type MagnitudeConfig = Schema.Schema.Type<typeof MagnitudeConfigSchema>

// =============================================================================
// Context limit policy defaults and helpers
// =============================================================================

export const DEFAULT_CONTEXT_LIMIT_POLICY = {
  softCapRatio: 0.9,
  softCapMaxTokens: 200_000,
} as const

export interface ResolvedContextLimitPolicy {
  readonly softCapRatio: number
  readonly softCapMaxTokens: number | null
}

export function resolveContextLimitPolicy(
  config: MagnitudeConfig
): ResolvedContextLimitPolicy {
  return {
    softCapRatio:
      config.contextLimits?.softCapRatio ??
      DEFAULT_CONTEXT_LIMIT_POLICY.softCapRatio,
    softCapMaxTokens: config.contextLimits?.softCapMaxTokens ?? null,
  }
}

export function computeContextLimits(
  hardCap: number,
  policy: ContextLimitPolicy
): { hardCap: number; softCap: number } {
  const softCapRatio =
    policy.softCapRatio ?? DEFAULT_CONTEXT_LIMIT_POLICY.softCapRatio
  const softCapMaxTokens = policy.softCapMaxTokens
  const ratioCap = Math.floor(hardCap * softCapRatio)
  const softCap =
    softCapMaxTokens == null ? ratioCap : Math.min(ratioCap, softCapMaxTokens)

  return { hardCap, softCap }
}
