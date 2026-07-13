/**
 * Retry/backoff constants and helpers for connection failures.
 *
 * Used by:
 *  - TurnProjection: computes `notBefore` timestamp on chain_continue triggers.
 *  - Cortex: enforces the cap by transforming the outcome before publishing.
 *  - RetryController: sleeps until the trigger's notBefore.
 *
 * All three must agree on the timing math, so it lives here.
 */

import { Schedule } from "effect"
import type { TurnOutcome } from '../events'

export const MAX_RETRIES = 5
export const BASE_DELAY_MS = 500
export const MAX_DELAY_MS = 30_000

/**
 * Exponential backoff with cap. attempt is 0-indexed.
 * Sequence: 500ms, 1s, 2s, 4s, 8s (capped at 30s for higher attempts).
 *
 * If a server-provided hint is present (Retry-After header), use the larger of
 * the two so we never retry sooner than the server told us.
 */
export function computeDelayMs(attempt: number, hintMs: number | undefined): number {
  const computed = Math.min(BASE_DELAY_MS * Math.pow(2, attempt), MAX_DELAY_MS)
  return hintMs !== undefined ? Math.max(hintMs, computed) : computed
}

/**
 * Extract the server's Retry-After hint from a ConnectionFailure outcome,
 * if any.
 */
export function getRetryAfterHint(outcome: TurnOutcome): number | undefined {
  if (outcome._tag !== 'ConnectionFailure') return undefined
  return outcome.detail.failure.retryAfterMs ?? undefined
}

export const TERMINAL_RETRY_EXHAUSTED_MESSAGE =
  'Lost connection to Magnitude. Check your network and try again.'

/**
 * Effect Schedule for connection-failure retries.
 * Exponential backoff from BASE_DELAY_MS, capped at MAX_DELAY_MS,
 * up to MAX_RETRIES attempts.
 *
 * Used by Autopilot, CompactionWorker, and any other non-fork workers
 * that need self-contained retries. Cortex uses computeDelayMs +
 * projection triggers instead.
 */
export const connectionRetrySchedule = Schedule.intersect(
  Schedule.recurs(MAX_RETRIES - 1),
  Schedule.exponential(`${BASE_DELAY_MS} millis`, 2).pipe(
    Schedule.either(Schedule.spaced(`${MAX_DELAY_MS} millis`)),
    Schedule.map((_) => undefined),
  ),
)
