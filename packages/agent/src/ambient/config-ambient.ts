import { Ambient } from '@magnitudedev/event-core'
import { Effect, Schema } from 'effect'
import { formatModelDisplayName } from '@magnitudedev/acn-protocol'
import {
  ModelFailureSchema,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
} from '@magnitudedev/acn-protocol'
import { type SlotId } from '@magnitudedev/roles'
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from '@magnitudedev/ai'
import {
  computeContextLimits,
  DEFAULT_CONTEXT_LIMIT_POLICY,
  type ResolvedContextLimitPolicy,
} from '@magnitudedev/storage'
import { ROLE_TO_SLOT, type RoleId } from '@magnitudedev/roles'

import { OUTPUT_TOKEN_RESERVE } from '../constants'

export const SlotConfigSchema = Schema.Struct({
  slotId: Schema.Literal('primary', 'secondary'),
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  modelDisplayName: Schema.String,
  profile: Schema.Struct({
    contextWindow: Schema.Number,
    maxOutputTokens: Schema.Number,
  }),
  vision: Schema.Boolean,
  hardCap: Schema.Number,
  softCap: Schema.Number,
  reasoningEffort: ReasoningEffortSchema,
  isUserOverride: Schema.Boolean,
  isFallback: Schema.Boolean,
})
export type SlotConfig = typeof SlotConfigSchema.Type

export const AgentSlotStateSchema = Schema.Union(
  Schema.TaggedStruct('Unassigned', {
    slotId: Schema.Literal('primary', 'secondary'),
  }),
  Schema.TaggedStruct('Pending', {
    slotId: Schema.Literal('primary', 'secondary'),
  }),
  Schema.TaggedStruct('Ready', {
    config: SlotConfigSchema,
  }),
  Schema.TaggedStruct('Unavailable', {
    slotId: Schema.Literal('primary', 'secondary'),
    failure: ModelFailureSchema,
  }),
)
export type AgentSlotState = typeof AgentSlotStateSchema.Type
export type UnassignedAgentSlot = Extract<AgentSlotState, { readonly _tag: 'Unassigned' }>
export type PendingAgentSlot = Extract<AgentSlotState, { readonly _tag: 'Pending' }>
export type ReadyAgentSlot = Extract<AgentSlotState, { readonly _tag: 'Ready' }>
export type UnavailableAgentSlot = Extract<AgentSlotState, { readonly _tag: 'Unavailable' }>

export const ConfigStateSchema = Schema.Struct({
  bySlot: Schema.Struct({
    primary: AgentSlotStateSchema,
    secondary: AgentSlotStateSchema,
  }),
})
export type ConfigState = typeof ConfigStateSchema.Type

const configStateEquivalent = Schema.equivalence(ConfigStateSchema)

export const sameConfigStateValue = (left: ConfigState, right: ConfigState): boolean =>
  configStateEquivalent(left, right)

export function getSlotConfig(state: ConfigState, slotId: SlotId): SlotConfig {
  const slot = state.bySlot[slotId]
  if (slot._tag !== 'Ready') throw new NoModelForSlotError(slotId)
  return slot.config
}

export function getSlotConfigOrNull(state: ConfigState, slotId: SlotId): SlotConfig | null {
  const slot = state.bySlot[slotId]
  return slot._tag === 'Ready' ? slot.config : null
}

export function getSlotConfigForRole(state: ConfigState, roleId: RoleId): SlotConfig {
  const slotId = ROLE_TO_SLOT[roleId]
  return getSlotConfig(state, slotId)
}

export function getSlotStateForRole(state: ConfigState, roleId: RoleId): AgentSlotState {
  return state.bySlot[ROLE_TO_SLOT[roleId]]
}

export class NoModelForSlotError extends Error {
  constructor(
    public readonly slotId: SlotId,
  ) {
    super(`No model available for slot ${slotId}. Check your API key and model configuration.`)
    this.name = 'NoModelForSlotError'
  }
}

export const ConfigAmbient = Ambient.define<ConfigState, never>({
  name: 'Config',
  initial: Effect.succeed({
    bySlot: {
      primary: { _tag: 'Pending', slotId: 'primary' },
      secondary: { _tag: 'Pending', slotId: 'secondary' },
    },
  }),
})

export function buildConfigStateFromSlots(
  catalogModels: readonly ProviderModelCatalogEntry[],
  slots: ModelSlotsState['slots'],
  policy: ResolvedContextLimitPolicy,
): ConfigState {
  const buildSlot = (slotId: SlotId): AgentSlotState => {
    const slot = slots[slotId]
    if (slot._tag === 'Unassigned') {
      return { _tag: 'Unassigned', slotId }
    }
    if (slot._tag === 'Resolving') {
      return { _tag: 'Pending', slotId }
    }
    if (slot.availability._tag === 'Pending') {
      return { _tag: 'Pending', slotId }
    }
    if (slot.availability._tag === 'Unavailable') {
      return { _tag: 'Unavailable', slotId, failure: slot.availability.failure }
    }
    const selectedModel = catalogModels.find((model) => model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)
    if (!selectedModel) {
      return {
        _tag: 'Unavailable',
        slotId,
        failure: {
          code: 'catalog_model_missing',
          message: 'The selected model is not available in the provider catalog.',
          retryable: false,
        },
      }
    }
    const hardCap = selectedModel.contextWindow - OUTPUT_TOKEN_RESERVE
    const { softCap } = computeContextLimits(hardCap, policy)
    return {
      _tag: 'Ready',
      config: {
        slotId,
        providerId: slot.selection.providerId,
        providerModelId: slot.selection.providerModelId,
        modelDisplayName: formatModelDisplayName(
          selectedModel.displayName,
          selectedModel.variantLabel,
        ),
        profile: {
          contextWindow: selectedModel.contextWindow,
          maxOutputTokens: selectedModel.maxOutputTokens,
        },
        vision: selectedModel.capabilities.vision,
        hardCap,
        softCap,
        reasoningEffort: slot.selection.reasoningEffort,
        isUserOverride: true,
        isFallback: false,
      },
    }
  }
  return {
    bySlot: {
      primary: buildSlot('primary'),
      secondary: buildSlot('secondary'),
    },
  }
}
