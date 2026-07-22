import { Ambient } from '@magnitudedev/event-core'
import { Effect } from 'effect'

import type { ModelSlotsState, ProviderModelCatalogEntry } from '@magnitudedev/sdk'
import { type SlotId } from '@magnitudedev/roles'
import { type ReasoningEffort } from '@magnitudedev/ai'
import {
  computeContextLimits,
  DEFAULT_CONTEXT_LIMIT_POLICY,
  type ResolvedContextLimitPolicy,
} from '@magnitudedev/storage'
import { ROLE_TO_SLOT, type RoleId } from '@magnitudedev/roles'

import { OUTPUT_TOKEN_RESERVE } from '../constants'

export interface SlotConfig {
  readonly slotId: SlotId
  readonly providerId: string
  readonly providerModelId: string
  readonly profile: { readonly contextWindow: number; readonly maxOutputTokens: number }
  readonly vision: boolean | undefined
  readonly hardCap: number
  readonly softCap: number
  readonly reasoningEffort: ReasoningEffort
  readonly isUserOverride: boolean
  readonly isFallback: boolean
}

export interface ReadyAgentSlot {
  readonly _tag: 'Ready'
  readonly config: SlotConfig
}

export interface UnavailableAgentSlot {
  readonly _tag: 'Unavailable'
  readonly slotId: SlotId
  readonly reason: string
}

export type AgentSlotState = ReadyAgentSlot | UnavailableAgentSlot

export interface ConfigState {
  readonly revision: number
  readonly bySlot: Readonly<Record<SlotId, AgentSlotState>>
  readonly catalogLoaded: boolean
}

export function getSlotConfig(state: ConfigState, slotId: SlotId): SlotConfig {
  const slot = state.bySlot[slotId]
  if (slot._tag === 'Unavailable') throw new NoModelForSlotError(slotId)
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
      primary: { _tag: 'Unavailable', slotId: 'primary', reason: 'not_loaded' },
      secondary: { _tag: 'Unavailable', slotId: 'secondary', reason: 'not_loaded' },
    },
    revision: 0,
    catalogLoaded: false,
  }),
})

export function buildConfigStateFromSlots(
  catalogModels: readonly ProviderModelCatalogEntry[],
  slots: ModelSlotsState['slots'],
  policy: ResolvedContextLimitPolicy,
  revision = 0,
): ConfigState {
  const buildSlot = (slotId: SlotId): AgentSlotState => {
    const slot = slots[slotId]
    if (slot._tag === 'Unassigned') {
      return { _tag: 'Unavailable', slotId, reason: slot._tag }
    }
    const selectedModel = catalogModels.find((model) => model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)
    if (!selectedModel) {
      return { _tag: 'Unavailable', slotId, reason: 'catalog_model_missing' }
    }
    const hardCap = selectedModel.contextWindow - OUTPUT_TOKEN_RESERVE
    const { softCap } = computeContextLimits(hardCap, policy)
    return {
      _tag: 'Ready',
      config: {
        slotId,
        providerId: slot.selection.providerId,
        providerModelId: slot.selection.providerModelId,
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
    revision,
    bySlot: {
      primary: buildSlot('primary'),
      secondary: buildSlot('secondary'),
    },
    catalogLoaded: true,
  }
}
