/**
 * RetryController Worker (Forked)
 *
 * Drives the retry-after-delay path for ConnectionFailure outcomes.
 *
 * Flow:
 *   1. Cortex publishes a turn_outcome with _tag='ConnectionFailure'.
 *   2. TurnProjection enqueues a chain_continue trigger with notBefore set to
 *      the future timestamp at which the next retry is allowed.
 *   3. This worker observes the same turn_outcome event, reads the trigger's
 *      notBefore from the projection, sleeps until that moment, then publishes
 *      a wake event. The wake causes onProjectionsSettled to re-evaluate, at
 *      which point TurnController sees the trigger is now due and fires
 *      turn_started.
 *
 * Cancellation:
 *   The worker is forked, so its sleep is automatically interrupted on user
 *   interrupt (via Worker.defineForked's built-in interrupt coordinator) and
 *   on agent_killed / worker_user_killed / worker_idle_closed via
 *   forkLifecycle.completeOn.
 *
 * The cap (MAX_RETRIES) is enforced in Cortex by transforming the outcome
 * before publishing — see workers/cortex.ts. This worker only ever sees
 * pre-cap ConnectionFailure outcomes, so it always sleeps and wakes.
 */

import { Effect, Duration, Option } from 'effect'
import { Worker } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { TurnProjection } from '../projections/turn'

export const RetryController = Worker.defineForked<AppEvent>()({
  name: 'RetryController',

  forkLifecycle: {
    activateOn: 'agent_created',
    completeOn: ['agent_killed', 'worker_user_killed', 'worker_idle_closed'],
  },

  eventHandlers: {
    turn_outcome: (event, publish, read) => Effect.gen(function* () {
      if (event.outcome._tag !== 'ConnectionFailure') return

      const turnFork = yield* read(TurnProjection, event.forkId)
      if (!turnFork) return

      // Find the chain_continue trigger the projection just enqueued for this
      // failure. If there isn't one with a notBefore, nothing to do.
      const pending = turnFork.triggers.find(
        (t) => t._tag === 'chain_continue' && Option.isSome(t.notBefore),
      )
      if (!pending || pending._tag !== 'chain_continue' || Option.isNone(pending.notBefore)) return

      const notBefore = pending.notBefore.value
      const delayMs = Math.max(0, notBefore - Date.now())
      logger.info(
        { forkId: event.forkId, delayMs, notBefore },
        '[RetryController] Scheduling retry wake',
      )

      if (delayMs > 0) {
        yield* Effect.sleep(Duration.millis(delayMs))
      }

      yield* publish({ type: 'wake', forkId: event.forkId })
    }).pipe(Effect.orDie),
  },
})
