/**
 * LifecycleCoordinator
 *
 * Owns session durability timing. This preserves the original persistence
 * policy: start the periodic flush after session initialization, flush shortly
 * after turn termination, and flush once more when the engine scope closes.
 *
 * Projection snapshots are saved as part of the same flush, using the cursor
 * returned by the event append.
 */

import { Cause, Context, Data, Effect, Layer, Ref, Stream } from 'effect'
import {
  EventBusCoreTag,
  EventSinkTag,
  HydrationContext,
  ProjectionSnapshotServiceTag,
} from '@magnitudedev/event-core'
import type { EventCursor, Timestamped } from '@magnitudedev/event-core'
import { logger } from '@magnitudedev/logger'

import type { AppEvent } from '../events'
import { ChatPersistence } from '../persistence/chat-persistence-service'
import { TurnProjection } from '../projections/turn'

const FLUSH_INTERVAL = '1500 millis'
const TURN_TERMINATED_FLUSH_DELAY = '100 millis'

type LifecycleFlushReason = 'timer' | 'turn_terminated' | 'shutdown'

class MissingEventCursor extends Data.TaggedError('MissingEventCursor')<{
  readonly eventCount: number
}> {}

export const LifecycleCoordinatorTag = Context.GenericTag<void>('LifecycleCoordinatorWorker')

export const LifecycleCoordinator = {
  Tag: LifecycleCoordinatorTag,
  Layer: Layer.scoped(
    LifecycleCoordinatorTag,
    Effect.gen(function* () {
      const workerScope = yield* Effect.scopeWith((scope) => Effect.succeed(scope))
      const eventBus = yield* EventBusCoreTag<AppEvent>()
      const eventSink = yield* EventSinkTag<AppEvent>()
      const persistence = yield* ChatPersistence
      const projectionSnapshots = yield* ProjectionSnapshotServiceTag
      const hydration = yield* HydrationContext
      const turnTerminatedPubSub = yield* TurnProjection.signals.turnTerminated.tag
      const timerStarted = yield* Ref.make(false)

      const requeueAndFail = <E>(pending: Timestamped<AppEvent>[], error: E) =>
        eventSink.prependEvents(pending).pipe(
          Effect.andThen(Effect.fail(error))
        )

      const saveProjectionSnapshot = (
        cursor: EventCursor,
        sessionId: string,
        reason: LifecycleFlushReason
      ) =>
        projectionSnapshots.captureProjectionSnapshot(cursor, sessionId).pipe(
          Effect.flatMap((snapshot) => persistence.saveProjectionSnapshot(snapshot)),
          Effect.catchAllCause((cause) =>
            Effect.sync(() => logger.warn({
              cause: Cause.pretty(cause),
              cursor,
              reason,
            }, '[LifecycleCoordinator] Projection snapshot save failed; event log remains authoritative'))
          )
        )

      const flushPendingEvents = (reason: LifecycleFlushReason) =>
        eventBus.checkpoint(Effect.gen(function* () {
          const pending = yield* eventSink.drainPending()
          if (pending.length === 0) return

          const metadata = yield* persistence.getSessionMetadata().pipe(
            Effect.catchAll((error) => requeueAndFail(pending, error))
          )

          const cursor = yield* persistence.persistNewEvents(pending).pipe(
            Effect.catchAll((error) => requeueAndFail(pending, error))
          )

          if (cursor === null) {
            yield* eventSink.prependEvents(pending)
            return yield* new MissingEventCursor({ eventCount: pending.length })
          }

          yield* saveProjectionSnapshot(cursor, metadata.sessionId, reason)
        }))

      const flushPendingEventsOrLog = (reason: LifecycleFlushReason) =>
        flushPendingEvents(reason).pipe(
          Effect.catchAllCause((cause) =>
            Effect.sync(() => logger.error({
              cause: Cause.pretty(cause),
              reason,
            }, '[LifecycleCoordinator] Failed to flush pending events'))
          ),
          Effect.asVoid
        )

      const startFlushTimer = Effect.gen(function* () {
        const alreadyStarted = yield* Ref.getAndSet(timerStarted, true)
        if (alreadyStarted) return

        yield* Effect.forkIn(
          Effect.forever(
            Effect.sleep(FLUSH_INTERVAL).pipe(
              Effect.andThen(flushPendingEventsOrLog('timer'))
            )
          ),
          workerScope
        )
      })

      // Registered before child fibers so this runs after they are interrupted.
      yield* Effect.acquireRelease(
        Effect.void,
        () => flushPendingEventsOrLog('shutdown')
      )

      yield* Effect.forkIn(
        Stream.runForEach(
          eventBus.subscribeToTypes(['session_initialized'] as const),
          () => startFlushTimer
        ),
        workerScope
      )

      yield* Effect.forkIn(
        Stream.runForEach(
          Stream.fromPubSub(turnTerminatedPubSub),
          () => Effect.gen(function* () {
            if (yield* hydration.isHydrating()) return
            yield* Effect.sleep(TURN_TERMINATED_FLUSH_DELAY)
            yield* flushPendingEventsOrLog('turn_terminated')
          })
        ),
        workerScope
      )
    })
  ),
} as const
