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

  eventHandlers: {
    session_initialized: (_event, _publish) => Effect.gen(function* () {
      yield* Effect.forkDaemon(
        Effect.repeat(
          flushPendingEvents,
          Schedule.spaced('1500 millis')
        )
      )
    }),
  },

  signalHandlers: (on) => [
    on(WorkingStateProjection.signals.forkBecameStable, ({ forkId }, publish) => Effect.gen(function* () {
      // Debounce: wait 100ms before persisting to batch multiple stable events
      yield* Effect.sleep('100 millis')
      yield* flushPendingEvents
    }))
  ]
})

const flushPendingEvents = Effect.gen(function* () {
  const eventSink = yield* EventSinkTag<AppEvent>()
  const persistence = yield* ChatPersistence

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
})
