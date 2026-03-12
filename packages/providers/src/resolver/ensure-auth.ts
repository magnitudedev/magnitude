/**
 * OAuth token refresh — ensures valid auth before inference calls.
 */

import { Effect } from 'effect'
import { logger } from '@magnitudedev/logger'
import { setAuth } from '../config'
import { setModel, getSlots } from '../state/provider-state'
import type { ModelSlot } from '../state/provider-state'
import { refreshAnthropicToken } from '../auth/anthropic-oauth'
import { refreshOpenAIToken } from '../auth/openai-oauth'
import { exchangeCopilotToken } from '../auth/copilot-oauth'
import type { OAuthAuth } from '../types'
import { AuthFailed } from '../errors/model-error'

/** Refresh OAuth tokens 5 minutes before actual expiry */
const TOKEN_EXPIRY_BUFFER_MS = 5 * 60_000

/**
 * Ensure the OAuth token for a slot is valid. If expired or expiring soon,
 * refresh it, persist the new tokens, and rebuild the ClientRegistry.
 *
 * No-op if auth is not OAuth or token is still valid.
 */
export function ensureAuth(slot: ModelSlot = 'primary'): Effect.Effect<void, AuthFailed> {
  return Effect.tryPromise({
    try: async () => {
      const s = getSlots()[slot]
      if (!s.auth || s.auth.type !== 'oauth' || !s.providerId || !s.modelId) return

      // Check if token is still valid (with buffer)
      if (s.auth.expiresAt > Date.now() + TOKEN_EXPIRY_BUFFER_MS) return

      logger.info(`[ProviderState] OAuth token expired or expiring for ${slot} slot, refreshing...`)

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
    },
    catch: (error) =>
      new AuthFailed({
        message: error instanceof Error ? error.message : 'Auth refresh failed',
      }),
  })
}