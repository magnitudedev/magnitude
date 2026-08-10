import { Option, Schema } from 'effect'
import { ProviderModelIdSchema } from '@magnitudedev/ai'
import {
  ModelOfferingTargetIdSchema,
  ProviderModelIdentitySchema,
  ModelServingConfigurationSchema,
  ModelPackageIdSchema,
  SlotIdSchema,
  SlotSelectionSchema,
  type ModelPackageId,
  type SlotId,
} from '@magnitudedev/acn-protocol'
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

// =============================================================================
// Slot-based model configuration
// =============================================================================

export { ModelPackageIdSchema, SlotIdSchema }
export type { ModelPackageId, SlotId }

export const SlotModelConfigSchema = SlotSelectionSchema
export type SlotModelConfig = Schema.Schema.Type<typeof SlotModelConfigSchema>

export const SlotSelectionStateSchema = Schema.Union(
  Schema.TaggedStruct('Unassigned', {}),
  Schema.TaggedStruct('Selected', {
    selection: SlotModelConfigSchema,
  }),
)
export type SlotSelectionState = Schema.Schema.Type<typeof SlotSelectionStateSchema>

export const unassignedSlotSelection = (): SlotSelectionState => ({ _tag: 'Unassigned' })
export const selectedSlotSelection = (selection: SlotModelConfig): SlotSelectionState => ({
  _tag: 'Selected',
  selection,
})
export const slotSelectionOption = (
  state: SlotSelectionState,
): Option.Option<SlotModelConfig> => state._tag === 'Selected'
  ? Option.some(state.selection)
  : Option.none()

export const PersistedLocalProviderOfferingSchema = Schema.Struct({
  providerModelId: ProviderModelIdSchema,
  targetId: ModelOfferingTargetIdSchema,
  configuration: ModelServingConfigurationSchema,
})
export type PersistedLocalProviderOffering =
  Schema.Schema.Type<typeof PersistedLocalProviderOfferingSchema>

export const ModelConfigSchema = Schema.Struct({
  slots: Schema.Struct({
    primary: SlotSelectionStateSchema,
    secondary: SlotSelectionStateSchema,
  }),
  localModelRecency: Schema.optionalWith(Schema.Struct({
    primary: Schema.Array(ProviderModelIdSchema),
    secondary: Schema.Array(ProviderModelIdSchema),
  }), {
    default: () => ({ primary: [], secondary: [] }),
  }),
  favoriteModels: Schema.optionalWith(
    Schema.Array(ProviderModelIdentitySchema),
    { default: () => [] },
  ),
  localProviderOfferings: Schema.optionalWith(
    Schema.Array(PersistedLocalProviderOfferingSchema),
    { default: () => [] },
  ),
  dismissedDownloadFailures: Schema.optionalWith(
    Schema.Array(ModelPackageIdSchema),
    { default: () => [] },
  ),
})
export type ModelConfig = Schema.Schema.Type<typeof ModelConfigSchema>

export const OnboardingConfigSchema = Schema.Struct({
  completed: Schema.Boolean,
})
export type OnboardingConfig = Schema.Schema.Type<typeof OnboardingConfigSchema>

export const MagnitudeConfigSchema = Schema.Struct({
  contextLimits: Schema.optional(ContextLimitPolicySchema),
  models: Schema.optional(ModelConfigSchema),
  onboarding: Schema.optionalWith(OnboardingConfigSchema, { as: 'Option', exact: true }),
  providers: SerializableOptional(CustomEndpointDeclarationsSchema),
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
