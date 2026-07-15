import { Schema } from 'effect'

const NullableOptional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(Schema.NullishOr(schema), {
    default: () => null as A | null,
  })

export const ContextLimitPolicySchema = Schema.Struct({
  softCapRatio: Schema.optional(Schema.Number),
  softCapMaxTokens: NullableOptional(Schema.Number),
})
export interface ContextLimitPolicy extends Omit<Schema.Schema.Type<typeof ContextLimitPolicySchema>, 'softCapMaxTokens'> {
  softCapMaxTokens: number | null
}

// =============================================================================
// Slot-based model configuration
// =============================================================================

/**
 * Slot identifiers. Defined locally because the storage package cannot import
 * from @magnitudedev/roles. Kept in sync with
 * `packages/roles/src/types.ts`.
 */
export const SlotIdSchema = Schema.Literal('primary', 'secondary')
export type SlotId = Schema.Schema.Type<typeof SlotIdSchema>

/**
 * Reasoning effort levels. Defined locally to avoid a cross-package dependency.
 * Kept in sync with `packages/providers/src/magnitude/contract.ts`.
 */
export const SlotModelConfigSchema = Schema.Struct({
  providerId: Schema.optional(Schema.String),
  providerModelId: Schema.optional(Schema.String),
  reasoningEffort: Schema.optional(Schema.String),
})
export type SlotModelConfig = Schema.Schema.Type<typeof SlotModelConfigSchema>

export const ModelConfigSchema = Schema.Struct({
  slots: Schema.optional(
    Schema.partial(Schema.Record({ key: SlotIdSchema, value: SlotModelConfigSchema }))
  ),
})
export type ModelConfig = Schema.Schema.Type<typeof ModelConfigSchema>

export const OnboardingConfigSchema = Schema.Struct({
  cliModelSetupVersion: Schema.optional(Schema.Number.pipe(Schema.int(), Schema.positive())),
  completedAt: Schema.optional(Schema.String),
})
export type OnboardingConfig = Schema.Schema.Type<typeof OnboardingConfigSchema>

export const MagnitudeConfigSchema = Schema.Struct({
  contextLimits: Schema.optional(ContextLimitPolicySchema),
  models: Schema.optional(ModelConfigSchema),
  onboarding: Schema.optional(OnboardingConfigSchema),
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
