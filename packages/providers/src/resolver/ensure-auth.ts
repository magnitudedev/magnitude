/**
 * OAuth token refresh — ensures valid auth before inference calls.
 */

import { Effect } from 'effect'
import { logger } from '@magnitudedev/logger'
import type { ModelSlot } from '../state/provider-state'
import { AuthFailed } from '../errors/model-error'
import { ProviderAuth, ProviderState } from '../runtime/contracts'

/** Refresh OAuth tokens 5 minutes before actual expiry */
const TOKEN_EXPIRY_BUFFER_MS = 5 * 60_000

/**
 * Ensure the OAuth token for a slot is valid. If expired or expiring soon,
 * refresh it, persist the new tokens, and rebuild the active runtime selection.
 *
 * No-op if auth is not OAuth or token is still valid.
 */
export function ensureAuth(
  slot: ModelSlot = 'primary',
): Effect.Effect<void, AuthFailed, ProviderAuth | ProviderState> {
  return Effect.gen(function* () {
    const auth = yield* ProviderAuth
    const state = yield* ProviderState

    const s = yield* state.getSlot(slot)
    if (!s.auth || s.auth.type !== 'oauth' || !s.providerId || !s.modelId) return

    if (s.auth.expiresAt > Date.now() + TOKEN_EXPIRY_BUFFER_MS) return

    logger.info(`[ProviderState] OAuth token expired or expiring for ${slot} slot, refreshing...`)

    const newAuth = yield* auth.refresh(s.providerId, s.auth.refreshToken).pipe(
      Effect.mapError((error) =>
        new AuthFailed({
          message: error instanceof Error ? error.message : 'Auth refresh failed',
        }),
      ),
    )

    if (!newAuth) {
      logger.warn('[ProviderState] OAuth refresh not supported for provider: ' + s.providerId)
      return
    }

    yield* auth.setAuth(s.providerId, newAuth)
    yield* state.setSelection(slot, s.providerId, s.modelId, newAuth, { persist: false })

    logger.info(`[ProviderState] OAuth token refreshed successfully for ${slot} slot`)
  })
}