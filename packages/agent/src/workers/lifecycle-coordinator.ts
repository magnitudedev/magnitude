/**
 * LifecycleCoordinator Worker
 *
 * Automatic persistence orchestrator.
 * Triggers persistence when any fork becomes stable (debounced 100ms).
 */

import { Effect, Schedule } from 'effect'
import { Worker, EventSinkTag } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'
import type { AppEvent } from '../events'
import { WorkingStateProjection } from '../projections/working-state'
import { ChatPersistence } from '../persistence/chat-persistence-service'

// =============================================================================
// Worker
// =============================================================================

export const LifecycleCoordinator = Worker.define<AppEvent>()({
  name: 'LifecycleCoordinator',

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ forkId }, publish) => Effect.gen(function* () {
      const eventSink = yield* EventSinkTag<AppEvent>()
      const persistence = yield* ChatPersistence

      // Debounce: wait 100ms before persisting to batch multiple stable events
      yield* Effect.sleep('100 millis')

      // Drain pending events
      const pending = yield* eventSink.drainPending()
      if (pending.length === 0) return

      // Persist with retry (3x, exponential backoff)
      yield* persistence.persistNewEvents(pending).pipe(
        Effect.retry({
          times: 3,
          schedule: Schedule.exponential('100 millis')
        }),
        Effect.catchAll((error) => Effect.gen(function* () {
          logger.error({
            context: 'LifecycleCoordinator',
            error: error instanceof Error ? error.stack ?? error.message : String(error),
            pendingCount: pending.length
          }, 'Persistence failed after retries, re-queuing events')
          yield* eventSink.prependEvents(pending)
        }))
      )
    }))
  ]
})
