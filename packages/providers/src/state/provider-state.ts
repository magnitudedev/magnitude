/**
 * Module-level singleton holding the current active provider/model state.
 *
 * Supports primary and secondary model slots. Each slot independently tracks
 * provider, model, auth, and cached ClientRegistry.
 *
 * The model proxy (createModelProxy) reads from here to route BAML calls.
 */

import type { ClientRegistry } from '@magnitudedev/llm-core'
import { logger } from '@magnitudedev/logger'
import { buildClientRegistry } from '../client-registry-builder'
import { getProvider } from '../registry'
import { setPrimarySelection, setBrowserSelection, loadConfig, saveConfig, getAuth, setAuth } from '../config'
import { detectProviders } from '../detect'
import { isBrowserCompatible } from '../browser-models'
import { initializeModels } from '../models-dev'
import { setLocalProviderConfig } from '../local-config'
import type { AuthInfo } from '../types'
import { DEFAULT_CONTEXT_WINDOW } from '../constants'
import { Model } from '../model/model'

export type ModelSlot = 'primary' | 'secondary' | 'browser'

// =============================================================================
// State — one set per slot
// =============================================================================

export interface SlotState {
  providerId: string | null
  modelId: string | null
  auth: AuthInfo | null
  registry: ClientRegistry | undefined
}

function emptySlot(): SlotState {
  return { providerId: null, modelId: null, auth: null, registry: undefined }
}

const slots: Record<ModelSlot, SlotState> = {
  primary: emptySlot(),
  secondary: emptySlot(),
  browser: emptySlot(),
}

/** Expose slots for auth-refresh and other internal modules */
export function getSlots(): Record<ModelSlot, SlotState> {
  return slots
}

// =============================================================================
// Resolve — the core abstraction
// =============================================================================


/**
 * Synchronously peek at a model slot, returning a Model and auth if configured.
 * Use this instead of resolveModel() from external consumers.
 */
export function peekSlot(slot: ModelSlot = 'primary'): { model: Model; auth: AuthInfo | null } | null {
  const s = slots[slot]
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
}

// =============================================================================
// Write operations
// =============================================================================

/**
 * Set the provider + model for a slot. Rebuilds the ClientRegistry.
 */
export function setModel(
  slot: ModelSlot,
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  persist = true,
): boolean {
  const provider = getProvider(providerId)
  if (!provider) {
    logger.warn(`[ProviderState] Unknown provider: ${providerId}`)
    return false
  }

  const registry = buildClientRegistry(providerId, modelId, auth)

  slots[slot].providerId = providerId
  slots[slot].modelId = modelId
  slots[slot].auth = auth
  slots[slot].registry = registry

  if (persist && slot === 'primary') {
    setPrimarySelection(providerId, modelId)
  }
  if (persist && slot === 'browser') {
    setBrowserSelection(providerId, modelId)
  }

  return true
}

/**
 * Clear a model slot.
 */
export function clearModel(slot: ModelSlot): void {
  slots[slot] = emptySlot()
}

/**
 * Get the context window size of a model slot.
 * Returns DEFAULT_CONTEXT_WINDOW if no model is selected or model doesn't specify one.
 */
export function getModelContextWindow(slot: ModelSlot = 'primary'): number {
  const s = slots[slot]
  if (!s.providerId || !s.modelId) return DEFAULT_CONTEXT_WINDOW
  const provider = getProvider(s.providerId)
  if (!provider) return DEFAULT_CONTEXT_WINDOW
  const model = provider.models.find(m => m.id === s.modelId)
  return model?.contextWindow ?? DEFAULT_CONTEXT_WINDOW
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

/**
 * Initialize provider state on startup.
 *
 * Priority:
 * 1. Stored config (primaryModel from ~/.magnitude/config.json)
 * 2. Auto-detected provider (env vars, stored auth)
 * 3. No override (uses BAML static client — requires ANTHROPIC_API_KEY)
 */
export async function initializeProviderState(): Promise<void> {
  // 0. Fetch/cache dynamic model lists from models.dev
  await initializeModels()

  // 1. Check stored config
  const config = loadConfig()

  // Always restore local provider config if it exists on disk
  const localOpts = config.providerOptions?.['local']
  if (localOpts?.baseUrl && localOpts?.modelId) {
    setLocalProviderConfig(localOpts.baseUrl, localOpts.modelId)
  }

  // Detect currently connected providers — used for validation
  const connectedProviders = detectProviders()
  const connectedIds = new Set(connectedProviders.map(d => d.provider.id))
  let configChanged = false

  // Primary model — restore from config if provider still connected, otherwise clear
  if (config.primaryModel) {
    if (connectedIds.has(config.primaryModel.providerId)) {
      const auth = getAuth(config.primaryModel.providerId)
      setModel('primary', config.primaryModel.providerId, config.primaryModel.modelId, auth ?? null, false)
    } else {
      config.primaryModel = null
      configChanged = true
    }
  }

  // Secondary model — restore from config if provider still connected, otherwise clear
  if (config.secondaryModel) {
    if (connectedIds.has(config.secondaryModel.providerId)) {
      const auth = getAuth(config.secondaryModel.providerId)
      setModel('secondary', config.secondaryModel.providerId, config.secondaryModel.modelId, auth ?? null, false)
    } else {
      config.secondaryModel = null
      configChanged = true
    }
  }

  // Browser model — restore from config if provider still connected and model is browser-compatible, otherwise clear
  if (config.browserModel) {
    if (connectedIds.has(config.browserModel.providerId) && isBrowserCompatible(config.browserModel.providerId, config.browserModel.modelId)) {
      const auth = getAuth(config.browserModel.providerId)
      setModel('browser', config.browserModel.providerId, config.browserModel.modelId, auth ?? null, false)
    } else {
      config.browserModel = null
      configChanged = true
    }
  }

  if (configChanged) saveConfig(config)
}

// =============================================================================
// Usage Tracking
// =============================================================================

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

function emptySlotUsage(): SlotUsage {
  return { inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0, inputCost: 0, outputCost: 0, totalCost: 0, callCount: 0 }
}

const slotUsage: Record<ModelSlot, SlotUsage> = {
  primary: emptySlotUsage(),
  secondary: emptySlotUsage(),
  browser: emptySlotUsage(),
}

/** Get cumulative usage for a slot */
export function getSlotUsage(slot: ModelSlot): SlotUsage {
  return { ...slotUsage[slot] }
}

/** Reset cumulative usage for a slot */
export function resetSlotUsage(slot: ModelSlot): void {
  slotUsage[slot] = emptySlotUsage()
}

/** Accumulate a call's usage into the slot total */
export function accumulateUsage(slot: ModelSlot, usage: CallUsage): void {
  const s = slotUsage[slot]
  s.callCount++
  if (usage.inputTokens !== null) s.inputTokens += usage.inputTokens
  if (usage.outputTokens !== null) s.outputTokens += usage.outputTokens
  if (usage.cacheReadTokens !== null) s.cacheReadTokens += usage.cacheReadTokens
  if (usage.cacheWriteTokens !== null) s.cacheWriteTokens += usage.cacheWriteTokens
  if (usage.inputCost !== null) s.inputCost += usage.inputCost
  if (usage.outputCost !== null) s.outputCost += usage.outputCost
  if (usage.totalCost !== null) s.totalCost += usage.totalCost
}