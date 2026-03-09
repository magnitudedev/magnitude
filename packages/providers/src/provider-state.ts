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
import { buildClientRegistry } from './client-registry-builder'
import { getProvider } from './registry'
import { setPrimarySelection, setBrowserSelection, loadConfig, saveConfig, getAuth, setAuth } from './config'
import { detectProviders } from './detect'
import { isBrowserCompatible } from './browser-models'
import { initializeModels } from './models-dev'
import { setLocalProviderConfig } from './local-config'
import type { AuthInfo, OAuthAuth } from './types'
import { refreshAnthropicToken } from './auth/anthropic-oauth'
import { refreshOpenAIToken } from './auth/openai-oauth'
import { exchangeCopilotToken } from './auth/copilot-oauth'
import { DEFAULT_CONTEXT_WINDOW } from './constants'

// =============================================================================
// Types
// =============================================================================

export type ModelSlot = 'primary' | 'secondary' | 'browser'

export interface ResolvedModel {
  providerId: string
  modelId: string
  auth: AuthInfo | null
  registry: ClientRegistry | undefined
  contextWindow: number
  maxOutputTokens: number | undefined
  isAnthropicOAuth: boolean
  isCodex: boolean
  isCopilotCodex: boolean
}

// =============================================================================
// Constants
// =============================================================================

/** Refresh OAuth tokens 5 minutes before actual expiry */
const TOKEN_EXPIRY_BUFFER_MS = 5 * 60_000

// =============================================================================
// State — one set per slot
// =============================================================================

interface SlotState {
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

// =============================================================================
// Resolve — the core abstraction
// =============================================================================

/**
 * Resolve a model slot to its full state. Returns null if the slot is not configured.
 */
export function resolveModel(slot: ModelSlot): ResolvedModel | null {
  const s = slots[slot]
  if (!s.providerId || !s.modelId) return null

  const provider = getProvider(s.providerId)
  const model = provider?.models.find(m => m.id === s.modelId)
  const contextWindow = model?.contextWindow ?? DEFAULT_CONTEXT_WINDOW

  return {
    providerId: s.providerId,
    modelId: s.modelId,
    auth: s.auth,
    registry: s.registry,
    contextWindow,
    maxOutputTokens: model?.maxOutputTokens,
    isAnthropicOAuth: s.providerId === 'anthropic' && s.auth?.type === 'oauth',
    isCodex: s.providerId === 'openai' && s.auth?.type === 'oauth',
    isCopilotCodex: s.providerId === 'github-copilot' && (s.modelId?.includes('codex') === true),
  }
}

// =============================================================================
// Read accessors (backward-compatible, delegate to primary slot)
// =============================================================================

export function getPrimaryProviderId(): string | null {
  return slots.primary.providerId
}

export function getPrimaryModelId(): string | null {
  return slots.primary.modelId
}

/** Get the ClientRegistry for the current primary provider/model. undefined = use static BAML fallback. */
export function getClientRegistry(): ClientRegistry | undefined {
  return slots.primary.registry
}

/** Check if current primary provider is Anthropic with OAuth (Pro/Max subscription). */
export function isAnthropicOAuth(): boolean {
  return slots.primary.providerId === 'anthropic' && slots.primary.auth?.type === 'oauth'
}

/** Check if current primary provider is OpenAI with OAuth (ChatGPT subscription → Codex endpoint). */
export function isOpenAICodex(): boolean {
  return slots.primary.providerId === 'openai' && slots.primary.auth?.type === 'oauth'
}

/** Check if current primary provider is GitHub Copilot with a Codex model selected. */
export function isCopilotCodex(): boolean {
  return slots.primary.providerId === 'github-copilot'
    && slots.primary.modelId?.includes('codex') === true
}

/** Get Codex auth credentials if primary is in Codex mode, else null. */
export function getCodexAuth(): { accessToken: string; accountId?: string } | null {
  if (!isOpenAICodex() || slots.primary.auth?.type !== 'oauth') return null
  return { accessToken: slots.primary.auth.accessToken, accountId: slots.primary.auth.accountId }
}

/** Get Copilot Codex auth credentials, else null. */
export function getCopilotCodexAuth(): { accessToken: string } | null {
  if (!isCopilotCodex() || slots.primary.auth?.type !== 'oauth') return null
  return { accessToken: slots.primary.auth.accessToken }
}

/** Check if current provider is Anthropic with OAuth (Pro/Max subscription). */
export function isAnthropicOAuthForSlot(slot: ModelSlot): boolean {
  return slots[slot].providerId === 'anthropic' && slots[slot].auth?.type === 'oauth'
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

/** Backward-compatible alias */
export function getPrimaryModelContextWindow(): number {
  return getModelContextWindow('primary')
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

/** Set the primary provider + model. Backward-compatible wrapper. */
export function setPrimaryModel(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  persist = true,
): boolean {
  return setModel('primary', providerId, modelId, auth, persist)
}

/** Set the secondary provider + model. */
export function setSecondaryModel(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  persist = true,
): boolean {
  return setModel('secondary', providerId, modelId, auth, persist)
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

/** Clear the primary provider — revert to BAML static default. */
export function clearPrimaryModel(): void {
  slots.primary = emptySlot()
}

/** Clear the secondary provider. */
export function clearSecondaryModel(): void {
  slots.secondary = emptySlot()
}

/** Set the browser provider + model. */
export function setBrowserModel(
  providerId: string,
  modelId: string,
  auth: AuthInfo | null,
  persist = true,
): boolean {
  return setModel('browser', providerId, modelId, auth, persist)
}

/** Clear the browser provider. */
export function clearBrowserModel(): void {
  slots.browser = emptySlot()
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
      setPrimaryModel(config.primaryModel.providerId, config.primaryModel.modelId, auth ?? null, false)
    } else {
      config.primaryModel = null
      configChanged = true
    }
  }

  // Secondary model — restore from config if provider still connected, otherwise clear
  if (config.secondaryModel) {
    if (connectedIds.has(config.secondaryModel.providerId)) {
      const auth = getAuth(config.secondaryModel.providerId)
      setSecondaryModel(config.secondaryModel.providerId, config.secondaryModel.modelId, auth ?? null, false)
    } else {
      config.secondaryModel = null
      configChanged = true
    }
  }

  // Browser model — restore from config if provider still connected and model is browser-compatible, otherwise clear
  if (config.browserModel) {
    if (connectedIds.has(config.browserModel.providerId) && isBrowserCompatible(config.browserModel.providerId, config.browserModel.modelId)) {
      const auth = getAuth(config.browserModel.providerId)
      setBrowserModel(config.browserModel.providerId, config.browserModel.modelId, auth ?? null, false)
    } else {
      config.browserModel = null
      configChanged = true
    }
  }

  if (configChanged) saveConfig(config)
}

/** Get a display summary of the current provider state. */
export function getProviderSummary(): { provider: string; model: string } | null {
  if (!slots.primary.providerId || !slots.primary.modelId) return null
  const provider = getProvider(slots.primary.providerId)
  const model = provider?.models.find(m => m.id === slots.primary.modelId)
  return {
    provider: provider?.name ?? slots.primary.providerId,
    model: model?.name ?? slots.primary.modelId,
  }
}

// =============================================================================
// OAuth Token Refresh
// =============================================================================

/**
 * Ensure the OAuth token for a slot is valid. If expired or expiring soon,
 * refresh it, persist the new tokens, and rebuild the ClientRegistry.
 *
 * No-op if auth is not OAuth or token is still valid.
 */
export async function ensureValidAuth(slot: ModelSlot = 'primary'): Promise<void> {
  const s = slots[slot]
  if (!s.auth || s.auth.type !== 'oauth' || !s.providerId || !s.modelId) return

  // Check if token is still valid (with buffer)
  if (s.auth.expiresAt > Date.now() + TOKEN_EXPIRY_BUFFER_MS) return

  logger.info(`[ProviderState] OAuth token expired or expiring for ${slot} slot, refreshing...`)

  try {
    let newAuth: OAuthAuth

    if (s.providerId === 'anthropic') {
      newAuth = await refreshAnthropicToken(s.auth.refreshToken)
    } else if (s.providerId === 'openai') {
      newAuth = await refreshOpenAIToken(s.auth.refreshToken)
    } else if (s.providerId === 'github-copilot') {
      newAuth = await exchangeCopilotToken(s.auth.refreshToken)
    } else {
      logger.warn('[ProviderState] OAuth refresh not supported for provider: ' + s.providerId)
      return
    }

    // Persist new tokens
    setAuth(s.providerId, newAuth)

    // Rebuild ClientRegistry with the new access token
    setModel(slot, s.providerId, s.modelId, newAuth, false)

    logger.info(`[ProviderState] OAuth token refreshed successfully for ${slot} slot`)
  } catch (error) {
    logger.error({ error }, `[ProviderState] OAuth token refresh failed for ${slot} slot, request will proceed with stale token`)
  }
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
