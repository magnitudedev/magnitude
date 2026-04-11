/**
 * EventBusCore
 *
 * Central event bus that coordinates event publishing:
 * - Delegates projection handling to ProjectionBus (two-phase processing)
 * - Broadcasts events to workers asynchronously
 * - Enforces hydration safety by skipping side-effects during replay
 *
 * Generic over E - the application's event union type.
 */

import { Effect, PubSub, Queue, Deferred, Stream, Context, Layer, Scope, Cause } from 'effect'
import { HydrationContext } from './hydration-context'
import { EventSinkTag, type EventSinkService } from './event-sink'
import { InterruptCoordinator } from './interrupt-coordinator'
import { ProjectionBusTag, type ProjectionBusService } from './projection-bus'
import { extractForkIdFromEvent } from '../worker/util'
import { FrameworkErrorReporter, FrameworkError, type FrameworkErrorReporterService } from './framework-error'

// Base constraint for events - must have a type discriminator
export interface BaseEvent {
  readonly type: string
  /**
   * Ephemeral events flow through the EventBus and trigger projections/signals,
   * but are NOT persisted to the EventSink and NOT replayed during hydration.
   *
   * Use this ONLY when:
   * 1. The notification originates outside a projection (so signals can't be used), AND
   * 2. The notification is derived/transient — persisting it would be wrong because
   *    the triggering code re-runs naturally on replay.
   *
   * If either condition isn't met, use a signal (condition 1) or a normal event (condition 2).
   */
  readonly ephemeral?: true
}

// Utility type: event with framework-added timestamp
export type Timestamped<E extends BaseEvent> = E & { readonly timestamp: number }

export interface EventBusCoreService<E extends BaseEvent> {
  publish: (event: E) => Effect.Effect<void>

  subscribeToTypes: <T extends E['type']>(
    types: readonly T[]
  ) => Stream.Stream<Timestamped<Extract<E, { type: T }>>>

  stream: Stream.Stream<Timestamped<E>>

  /**
   * Eagerly subscribe to the event bus. Returns an Effect that creates the
   * subscription immediately (buffering starts now) and yields a Stream.
   * Use this when you need events that may be published between subscription
   * time and stream consumption time (e.g., spawning a fork fiber).
   * The subscription is scoped — it will be cleaned up when the scope closes.
   */
  subscribe: () => Effect.Effect<Stream.Stream<Timestamped<E>>, never, Scope.Scope>
}

// Create a tag for a specific event type E
export const EventBusCoreTag = <E extends BaseEvent>() =>
  Context.GenericTag<EventBusCoreService<E>>('EventBusCore')

// Type helper
export type EventBusCoreTagType<E extends BaseEvent> = ReturnType<typeof EventBusCoreTag<E>>

// Create the layer for a specific event type E
export function makeEventBusCoreLayer<E extends BaseEvent>(): Layer.Layer<
  EventBusCoreService<E>,
  never,
  HydrationContext | EventSinkService<E> | InterruptCoordinator | ProjectionBusService<E> | FrameworkErrorReporterService
> {
  const Tag = EventBusCoreTag<E>()
  const SinkTag = EventSinkTag<E>()
  const ProjBusTag = ProjectionBusTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const hydration = yield* HydrationContext
    const sink = yield* SinkTag
    const interruptCoordinator = yield* InterruptCoordinator
    const projectionBus = yield* ProjBusTag
    const reporter = yield* FrameworkErrorReporter
    const pubsub = yield* PubSub.unbounded<Timestamped<E>>()
    yield* interruptCoordinator.beginExecution(null)

    // Sequential event processing queue.
    // Events are enqueued by publish() and processed one at a time by a
    // single consumer fiber, guaranteeing that all projection updates and
    // signal propagation for event A complete before event B begins.
    const eventQueue = yield* Queue.unbounded<{ event: Timestamped<E>; done: Deferred.Deferred<void, Cause.Cause<never>> }>()

    // Consumer fiber — processes events sequentially
    yield* Effect.forkScoped(
      Effect.forever(
        Effect.gen(function* () {
          const { event, done } = yield* Queue.take(eventQueue)
          yield* Effect.gen(function* () {
            yield* projectionBus.processEvent(event)

            if (yield* hydration.isHydrating()) {
              return
            }

            if (event.type === 'interrupt') {
              yield* interruptCoordinator.interrupt(extractForkIdFromEvent(event))
            }

            if (!event.ephemeral) {
              yield* sink.append(event).pipe(
                Effect.catchAllCause((cause) =>
                  reporter.report(FrameworkError.SinkError({ eventType: event.type, cause }))
                )
              )
            }
            yield* PubSub.publish(pubsub, event).pipe(
              Effect.catchAllCause((cause) =>
                reporter.report(FrameworkError.BroadcastError({ eventType: event.type, cause }))
              )
            )
          }).pipe(
            Effect.matchCauseEffect({
              // Only truly fatal infrastructure errors reach here now
              onFailure: (cause) =>
                Effect.logError(`[EventBus] Critical failure for: ${event.type}`, cause).pipe(
                  Effect.andThen(Deferred.fail(done, cause))
                ),
              onSuccess: () => Deferred.succeed(done, undefined)
            })
          )
        })
      )
    )

    return {
      publish: (event: E) => Effect.gen(function* () {
        const timestamp = 'timestamp' in event && typeof event.timestamp === 'number'
          ? event.timestamp
          : Date.now()
        const timestamped: Timestamped<E> = Object.assign({}, event, { timestamp })
        const done = yield* Deferred.make<void, Cause.Cause<never>>()
        yield* Queue.offer(eventQueue, { event: timestamped, done })
        yield* Deferred.await(done).pipe(
          Effect.catchAll((cause) => Effect.failCause(cause))
        )
      }),

      subscribeToTypes: <T extends E['type']>(types: readonly T[]) =>
        Stream.fromPubSub(pubsub).pipe(
          Stream.filter((e): e is Timestamped<Extract<E, { type: T }>> =>
            types.some(t => t === e.type)
          )
        ),

      stream: Stream.fromPubSub(pubsub),

      subscribe: () => Effect.map(
        PubSub.subscribe(pubsub),
        (queue) => Stream.fromQueue(queue) as Stream.Stream<Timestamped<E>>
      )
    }
  }))
}
