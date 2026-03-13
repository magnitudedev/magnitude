/**
 * OAuth token refresh — ensures valid auth before inference calls.
 *
 * Handles cross-process and same-process concurrency:
 * - Re-reads auth from disk before refreshing (another session may have already refreshed)
 * - Uses a per-provider file lock to prevent concurrent refresh token rotation
 * - Retries with disk re-read on failure as a final safety net
 */

import * as fs from 'fs'
import * as path from 'path'
import { Effect } from 'effect'
import { logger } from '@magnitudedev/logger'
import type { ModelSlot } from '../state/provider-state'
import { AuthFailed } from '../errors/model-error'
import { ProviderAuth, ProviderState } from '../runtime/contracts'
import { getAuth, setAuth } from '../config'
import { refreshAnthropicToken } from '../auth/anthropic-oauth'
import { refreshOpenAIToken } from '../auth/openai-oauth'
import { exchangeCopilotToken } from '../auth/copilot-oauth'
import type { AuthInfo } from '../types'

/** Refresh OAuth tokens 5 minutes before actual expiry */
const TOKEN_EXPIRY_BUFFER_MS = 5 * 60_000
const LOCK_DIR = path.join(process.env.HOME ?? '~', '.magnitude')
const LOCK_STALE_MS = 30_000
const LOCK_RETRY_DELAY_MS = 200
const LOCK_MAX_ATTEMPTS = 50

function isFreshOAuthAuth(auth: AuthInfo | null | undefined): auth is Extract<AuthInfo, { type: 'oauth' }> {
  return auth?.type === 'oauth' && auth.expiresAt > Date.now() + TOKEN_EXPIRY_BUFFER_MS
}

function getOAuthRefreshToken(auth: AuthInfo | null | undefined): string | null {
  return auth?.type === 'oauth' ? auth.refreshToken : null
}

async function refreshForProvider(providerId: string, refreshToken: string) {
  if (providerId === 'anthropic') return refreshAnthropicToken(refreshToken)
  if (providerId === 'openai') return refreshOpenAIToken(refreshToken)
  if (providerId === 'github-copilot') return exchangeCopilotToken(refreshToken)
  return null
}

async function withProviderLock<T>(providerId: string, fn: () => Promise<T>): Promise<T> {
  fs.mkdirSync(LOCK_DIR, { recursive: true })

  const lockPath = path.join(LOCK_DIR, `auth-refresh-${providerId}.lock`)
  let acquired = false

  for (let attempt = 0; attempt < LOCK_MAX_ATTEMPTS; attempt++) {
    try {
      fs.mkdirSync(lockPath)
      acquired = true
      break
    } catch (error) {
      const err = error as NodeJS.ErrnoException
      if (err.code !== 'EEXIST') {
        throw error
      }

      try {
        const stat = fs.statSync(lockPath)
        if (Date.now() - stat.mtimeMs > LOCK_STALE_MS) {
          fs.rmdirSync(lockPath)
          continue
        }
      } catch {
        continue
      }

      await new Promise((resolve) => setTimeout(resolve, LOCK_RETRY_DELAY_MS))
    }
  }

  if (!acquired) {
    logger.warn(`[ensureAuth] Timed out waiting for refresh lock for ${providerId}, proceeding without lock`)
  }

  try {
    return await fn()
  } finally {
    if (acquired) {
      try {
        fs.rmdirSync(lockPath)
      } catch {
        // ignore lock cleanup failures
      }
    }
  }
}

/**
 * Ensure the OAuth token for a slot is valid. If expired or expiring soon,
 * refresh it, persist the new tokens, and rebuild the active runtime selection.
 *
 * No-op if auth is not OAuth or token is still valid.
 *
 * Flow:
 * 1. Check in-memory token — if fresh, return immediately
 * 2. Re-read from disk — if another session already refreshed, use that token
 * 3. Acquire per-provider file lock (~/.magnitude/auth-refresh-{providerId}.lock)
 * 4. Re-check disk inside lock — we may have waited behind another process that already refreshed
 * 5. Refresh using disk's refresh token, persist while holding lock
 * 6. On failure, re-read disk one more time (narrow race safety net)
 */
export function ensureAuth(
  slot: ModelSlot = 'primary',
): Effect.Effect<void, AuthFailed, ProviderAuth | ProviderState> {
  return Effect.gen(function* () {
    const auth = yield* ProviderAuth
    const state = yield* ProviderState

    const s = yield* state.getSlot(slot)
    if (!s.auth || s.auth.type !== 'oauth' || !s.providerId || !s.modelId) return

    const currentAuth = s.auth
    if (currentAuth.expiresAt > Date.now() + TOKEN_EXPIRY_BUFFER_MS) return

    const { providerId, modelId } = s

    logger.info(`[ProviderState] OAuth token expired or expiring for ${slot} slot, refreshing...`)

    const diskAuth = yield* auth.getAuth(providerId)
    if (isFreshOAuthAuth(diskAuth)) {
      logger.info(`[ensureAuth] Found fresh token on disk for ${providerId}, skipping refresh`)
      yield* state.setSelection(slot, providerId, modelId, diskAuth, { persist: false })
      return
    }

    const result = yield* Effect.tryPromise({
      try: () =>
        withProviderLock(providerId, async () => {
          const lockedDiskAuth = getAuth(providerId)
          if (isFreshOAuthAuth(lockedDiskAuth)) {
            return { action: 'use-disk', auth: lockedDiskAuth } as const
          }

          const refreshToken = getOAuthRefreshToken(lockedDiskAuth) ?? currentAuth.refreshToken

          try {
            const newAuth = await refreshForProvider(providerId, refreshToken)
            if (!newAuth) {
              return { action: 'unsupported' } as const
            }

            setAuth(providerId, newAuth)
            return { action: 'refreshed', auth: newAuth } as const
          } catch (error) {
            const retryDiskAuth = getAuth(providerId)
            if (isFreshOAuthAuth(retryDiskAuth)) {
              return { action: 'use-disk', auth: retryDiskAuth } as const
            }
            throw error
          }
        }),
      catch: (error) =>
        new AuthFailed({
          message: error instanceof Error ? error.message : 'Auth refresh failed',
        }),
    })

    if (result.action === 'unsupported') {
      logger.warn('[ProviderState] OAuth refresh not supported for provider: ' + providerId)
      return
    }

    yield* state.setSelection(slot, providerId, modelId, result.auth, { persist: false })

    logger.info(
      result.action === 'refreshed'
        ? `[ProviderState] OAuth token refreshed successfully for ${slot} slot`
        : `[ensureAuth] Using refreshed disk token for ${slot} slot`,
    )
  })
}