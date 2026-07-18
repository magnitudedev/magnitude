import { Schema } from 'effect'
import { ReasoningEffortSchema } from '@magnitudedev/ai'

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
  reasoningEffort: Schema.optional(ReasoningEffortSchema),
})
export type SlotModelConfig = Schema.Schema.Type<typeof SlotModelConfigSchema>

export const ModelConfigSchema = Schema.Struct({
  slots: Schema.optional(
    Schema.partial(Schema.Record({ key: SlotIdSchema, value: SlotModelConfigSchema }))
  ),
  localSlotIntent: Schema.optional(
    Schema.partial(Schema.Record({ key: SlotIdSchema, value: Schema.Literal('local', 'cloud') }))
  ),
  localModelRecency: Schema.optional(
    Schema.partial(Schema.Record({ key: SlotIdSchema, value: Schema.Array(Schema.String) }))
  ),
})
export type ModelConfig = Schema.Schema.Type<typeof ModelConfigSchema>

export const OnboardingFlowIdSchema = Schema.Literal('model_setup')
export type OnboardingFlowId = Schema.Schema.Type<typeof OnboardingFlowIdSchema>

export const OnboardingCompletionSchema = Schema.Struct({
  version: Schema.Number.pipe(Schema.int(), Schema.positive()),
  completedAt: Schema.String,
})
export type OnboardingCompletion = Schema.Schema.Type<typeof OnboardingCompletionSchema>

export const OnboardingConfigSchema = Schema.Struct({
  completions: Schema.optional(
    Schema.partial(Schema.Record({ key: OnboardingFlowIdSchema, value: OnboardingCompletionSchema })),
  ),
})
export type OnboardingConfig = Schema.Schema.Type<typeof OnboardingConfigSchema>

export const LocalInferenceUsageSelectionSchema = Schema.Struct({
  sessionConcurrency: Schema.Literal('one', 'up_to_three'),
})
export type LocalInferenceUsageSelection = Schema.Schema.Type<typeof LocalInferenceUsageSelectionSchema>

export const DurableLocalModelBindingSchema = Schema.Union(
  Schema.TaggedStruct('Managed', {
    selectionId: Schema.String,
    artifactId: Schema.String,
    providerModelId: Schema.String,
    contextTokens: Schema.Number.pipe(Schema.int(), Schema.positive()),
    parallelSlots: Schema.Number.pipe(Schema.int(), Schema.positive()),
  }),
  Schema.TaggedStruct('External', {
    selectionId: Schema.String,
    endpointConfigId: Schema.String,
    providerModelId: Schema.String,
    contextTokens: Schema.Number.pipe(Schema.int(), Schema.positive()),
  }),
)
export type DurableLocalModelBinding = Schema.Schema.Type<typeof DurableLocalModelBindingSchema>

export const SelectedLocalModelProfileSchema = Schema.Struct({
  configurationId: Schema.String,
  catalogModelId: Schema.String,
  contextTokens: Schema.Number.pipe(Schema.int(), Schema.positive()),
  parallelSlots: Schema.Number.pipe(Schema.int(), Schema.positive()),
})
export type SelectedLocalModelProfile = Schema.Schema.Type<typeof SelectedLocalModelProfileSchema>

export const LocalInferenceConfigSchema = Schema.Struct({
  usage: Schema.optional(LocalInferenceUsageSelectionSchema),
  selectedProfile: Schema.optional(SelectedLocalModelProfileSchema),
  binding: Schema.optional(DurableLocalModelBindingSchema),
})
export type LocalInferenceConfig = Schema.Schema.Type<typeof LocalInferenceConfigSchema>

export const MagnitudeConfigSchema = Schema.Struct({
  contextLimits: Schema.optional(ContextLimitPolicySchema),
  models: Schema.optional(ModelConfigSchema),
  onboarding: Schema.optional(OnboardingConfigSchema),
  localInference: Schema.optional(LocalInferenceConfigSchema),
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
