/**
 * Provider Client — authenticated API access for a model slot.
 *
 * Lower-level abstraction that provides auth, endpoint routing, model resolution,
 * reasoning effort, and usage tracking without prescribing request/response shapes.
 *
 * Used by:
 * - model-proxy.ts (for its Responses API streaming path)
 * - native-openai strategy (for structured function-calling requests)
 */

import {
  resolveModel, ensureValidAuth, accumulateUsage,
  type ModelSlot, type ResolvedModel, type CallUsage,
} from './provider-state'
import { buildUsage } from './usage'
import { getCodexReasoningEffort } from './reasoning-effort'
import { COPILOT_HEADERS } from './auth/copilot-oauth'

// =============================================================================
// Endpoints
// =============================================================================

const CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'
const COPILOT_BASE_URL = 'https://api.githubcopilot.com/v1'

// =============================================================================
// ProviderClient Interface
// =============================================================================

export interface ProviderClient {
  readonly slot: ModelSlot

  /** Resolve current model state. Null if slot is unconfigured. */
  resolve(): ResolvedModel | null

  /** Ensure OAuth token is fresh. No-op for API key auth. */
  ensureAuth(): Promise<void>

  /**
   * Get the API base URL for the current provider/auth combination.
   * - OpenAI OAuth (Codex): 'https://chatgpt.com/backend-api/codex'
   * - GitHub Copilot Codex: 'https://api.githubcopilot.com/v1'
   * - Otherwise: undefined (use SDK/provider default)
   */
  getBaseURL(): string | undefined

  /**
   * Get the Responses API endpoint URL (base + /responses).
   * For raw fetch calls that bypass the OpenAI SDK.
   */
  getResponsesEndpoint(): string

  /**
   * Get authenticated headers for HTTP calls.
   * Includes Authorization, ChatGPT-Account-Id, Copilot headers as applicable.
   */
  getHeaders(): Record<string, string>

  /** Get the resolved model ID (e.g., 'gpt-5.3-codex'). */
  getModelId(): string | null

  /**
   * Get reasoning effort config to merge into the request body.
   * Returns null if no override is needed for this model.
   */
  getReasoningEffort(): Record<string, unknown> | null

  /**
   * Build a CallUsage from raw token counts, calculate costs,
   * and accumulate into the slot's session totals.
   */
  recordUsage(raw: {
    input_tokens?: number
    output_tokens?: number
    cache_read_tokens?: number
    cache_write_tokens?: number
  }): CallUsage
}

// =============================================================================
// Factory
// =============================================================================

export function createProviderClient(slot: ModelSlot): ProviderClient {
  return {
    slot,

    resolve() {
      return resolveModel(slot)
    },

    async ensureAuth() {
      await ensureValidAuth(slot)
    },

    getBaseURL() {
      const resolved = resolveModel(slot)
      if (!resolved) return undefined
      if (resolved.isCodex) return CODEX_BASE_URL
      if (resolved.isCopilotCodex) return COPILOT_BASE_URL
      return undefined
    },

    getResponsesEndpoint() {
      const resolved = resolveModel(slot)
      if (!resolved) return 'https://api.openai.com/v1/responses'
      if (resolved.isCodex) return CODEX_BASE_URL + '/responses'
      if (resolved.isCopilotCodex) return COPILOT_BASE_URL + '/responses'
      return 'https://api.openai.com/v1/responses'
    },

    getHeaders() {
      const resolved = resolveModel(slot)
      if (!resolved) return {}

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      }

      if (resolved.auth?.type === 'oauth') {
        headers['Authorization'] = `Bearer ${resolved.auth.accessToken}`
        if (resolved.isCodex && resolved.auth.accountId) {
          headers['ChatGPT-Account-Id'] = resolved.auth.accountId
        }
        if (resolved.isCopilotCodex) {
          headers['Openai-Intent'] = 'conversation-edits'
          headers['x-initiator'] = 'user'
          Object.assign(headers, COPILOT_HEADERS)
        }
      } else if (resolved.auth?.type === 'api') {
        headers['Authorization'] = `Bearer ${resolved.auth.key}`
      } else {
        const envKey = process.env.OPENAI_API_KEY
        if (envKey) {
          headers['Authorization'] = `Bearer ${envKey}`
        }
      }

      return headers
    },

    getModelId() {
      return resolveModel(slot)?.modelId ?? null
    },

    getReasoningEffort() {
      const resolved = resolveModel(slot)
      if (!resolved) return null
      const effort = getCodexReasoningEffort(resolved.modelId)
      if (!effort) return null
      return { reasoning: { effort } }
    },

    recordUsage(raw) {
      const resolved = resolveModel(slot)
      const usage = buildUsage(
        resolved,
        raw.input_tokens ?? null,
        raw.output_tokens ?? null,
        raw.cache_read_tokens ?? null,
        raw.cache_write_tokens ?? null,
      )
      accumulateUsage(slot, usage)
      return usage
    },
  }
}
