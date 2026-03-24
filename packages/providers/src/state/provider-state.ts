/**
 * Module-level singleton holding the current active provider/model state.
 *
 * The model proxy (createModelProxy) reads from here to route BAML calls.
 */

import type { ClientRegistry } from '@magnitudedev/llm-core'
import type { ProviderOptions } from '@magnitudedev/storage'
import { logger } from '@magnitudedev/logger'
import { buildClientRegistry } from '../client-registry-builder'
import { getProvider } from '../registry'
import type { AuthInfo } from '../types'
import { DEFAULT_CONTEXT_WINDOW } from '../constants'
import { Model } from '../model/model'

export interface SlotState {
  providerId: string | null
  modelId: string | null
  auth: AuthInfo | null
  registry: ClientRegistry | undefined
}

export interface CallUsage {
  inputTokens: number | null
  outputTokens: number | null
  cacheReadTokens: number | null
  cacheWriteTokens: number | null
  inputCost: number | null
  outputCost: number | null
  totalCost: number | null
}

export interface SlotUsage {
  inputTokens: number
  outputTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  inputCost: number
  outputCost: number
  totalCost: number
  callCount: number
}

export interface ProviderStateStore<TSlot extends string> {
  readonly getSlots: () => Map<TSlot, SlotState>
  readonly peekSlot: (slot: TSlot) => { model: Model; auth: AuthInfo | null } | null
  readonly setModel: (
    slot: TSlot,
    providerId: string,
    modelId: string,
    auth: AuthInfo | null,
    providerOptions?: ProviderOptions,
  ) => boolean
  readonly clearModel: (slot: TSlot) => void
  readonly getModelContextWindow: (slot: TSlot) => number
  readonly getSlotUsage: (slot: TSlot) => SlotUsage
  readonly resetSlotUsage: (slot: TSlot) => void
  readonly accumulateUsage: (slot: TSlot, usage: CallUsage) => void
}

function emptySlot(): SlotState {
  return { providerId: null, modelId: null, auth: null, registry: undefined }
}

function emptySlotUsage(): SlotUsage {
  return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, inputCost: 0, outputCost: 0, totalCost: 0, callCount: 0 }
}

export function makeProviderStateStore<TSlot extends string>(): ProviderStateStore<TSlot> {
  const slots = new Map<TSlot, SlotState>()
  const slotUsage = new Map<TSlot, SlotUsage>()

  const getOrCreateSlot = (slot: TSlot): SlotState => {
    const existing = slots.get(slot)
    if (existing) return existing
    const created = emptySlot()
    slots.set(slot, created)
    return created
  }

  const getOrCreateSlotUsage = (slot: TSlot): SlotUsage => {
    const existing = slotUsage.get(slot)
    if (existing) return existing
    const created = emptySlotUsage()
    slotUsage.set(slot, created)
    return created
  }

  return {
    getSlots: () => slots,
    peekSlot: (slot) => {
      const s = getOrCreateSlot(slot)
      if (!s.providerId || !s.modelId) return null
      const provider = getProvider(s.providerId)
      const def = provider?.models.find(m => m.id === s.modelId)
      return {
        model: new Model({
          id: s.modelId,
          providerId: s.providerId,
          name: def?.name ?? s.modelId,
          contextWindow: def?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
          maxOutputTokens: def?.maxOutputTokens ?? null,
          costs: def?.cost ? {
            inputPerM: def.cost.input,
            outputPerM: def.cost.output,
            cacheReadPerM: def.cost.cache_read ?? null,
            cacheWritePerM: def.cost.cache_write ?? null,
          } : null,
        }),
        auth: s.auth,
      }
    },
    setModel: (slot, providerId, modelId, auth, providerOptions) => {
      const provider = getProvider(providerId)
      if (!provider) {
        logger.warn(`[ProviderState] Unknown provider: ${providerId}`)
        return false
      }

      const registry = buildClientRegistry(providerId, modelId, auth, providerOptions)
      const slotState = getOrCreateSlot(slot)
      slotState.providerId = providerId
      slotState.modelId = modelId
      slotState.auth = auth
      slotState.registry = registry

      return true
    },
    clearModel: (slot) => {
      slots.set(slot, emptySlot())
    },
    getModelContextWindow: (slot) => {
      const s = getOrCreateSlot(slot)
      if (!s.providerId || !s.modelId) return DEFAULT_CONTEXT_WINDOW
      const provider = getProvider(s.providerId)
      if (!provider) return DEFAULT_CONTEXT_WINDOW
      const model = provider.models.find(m => m.id === s.modelId)
      return model?.contextWindow ?? DEFAULT_CONTEXT_WINDOW
    },
    getSlotUsage: (slot) => ({ ...getOrCreateSlotUsage(slot) }),
    resetSlotUsage: (slot) => {
      slotUsage.set(slot, emptySlotUsage())
    },
    accumulateUsage: (slot, usage) => {
      const s = getOrCreateSlotUsage(slot)
      s.callCount++
      if (usage.inputTokens !== null) s.inputTokens += usage.inputTokens
      if (usage.outputTokens !== null) s.outputTokens += usage.outputTokens
      if (usage.cacheReadTokens !== null) s.cacheReadTokens += usage.cacheReadTokens
      if (usage.cacheWriteTokens !== null) s.cacheWriteTokens += usage.cacheWriteTokens
      if (usage.inputCost !== null) s.inputCost += usage.inputCost
      if (usage.outputCost !== null) s.outputCost += usage.outputCost
      if (usage.totalCost !== null) s.totalCost += usage.totalCost
    },
  }
}

/**
 * Check if switching to a given model is safe given the current token usage.
 * Returns an error message if the switch would be invalid, or null if safe.
 */
export function validateModelSwitch(providerId: string, modelId: string, currentTokenEstimate: number): string | null {
  const provider = getProvider(providerId)
  if (!provider) return null
  const model = provider.models.find(m => m.id === modelId)
  const contextWindow = model?.contextWindow ?? DEFAULT_CONTEXT_WINDOW
  if (currentTokenEstimate > contextWindow) {
    const formatTokens = (n: number) => {
      if (n >= 1000) {
        const v = (n / 1000).toFixed(1)
        return (v.endsWith('.0') ? v.slice(0, -2) : v) + 'k'
      }
      return String(n)
    }
    return `Cannot switch to ${model?.name ?? modelId} — current context usage (${formatTokens(currentTokenEstimate)} tokens) exceeds its context window (${formatTokens(contextWindow)} tokens)`
  }
  return null
}