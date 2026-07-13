/**
 * EventSink
 *
 * Simple accumulator for events in memory.
 * Used for persistence - provides all events to write to database.
 *
 * Generic over E - the application's event union type.
 * Follows the same pattern as WorkerBus (GenericTag + layer factory).
 */

import { Effect, Ref, Context, Layer } from 'effect'
import type { BaseEvent, Timestamped } from './event-bus-core'

export interface EventSinkService<E extends BaseEvent = BaseEvent> {
  append: (event: Timestamped<E>) => Effect.Effect<void>
  readPending: () => Effect.Effect<Timestamped<E>[]>
  drainPending: () => Effect.Effect<Timestamped<E>[]>
  prependEvents: (events: Timestamped<E>[]) => Effect.Effect<void>
}

export const EventSinkTag = <E extends BaseEvent>() =>
  Context.GenericTag<EventSinkService<E>>('EventSink')

export function makeEventSinkLayer<E extends BaseEvent>(): Layer.Layer<EventSinkService<E>> {
  const Tag = EventSinkTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const pendingRef = yield* Ref.make<Timestamped<E>[]>([])

    return {
      append: (event: Timestamped<E>) =>
        Ref.update(pendingRef, events => [...events, event]),

      readPending: () =>
        Ref.get(pendingRef),

      drainPending: () =>
        Ref.getAndSet(pendingRef, []),

      prependEvents: (events: Timestamped<E>[]) =>
        Ref.update(pendingRef, pending => [...events, ...pending])
    }
  }))
}
