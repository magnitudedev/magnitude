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
import type { BaseEvent } from './event-bus-core'

export interface EventSinkService<E extends BaseEvent = BaseEvent> {
  append: (event: E) => Effect.Effect<void>
  readPending: () => Effect.Effect<E[]>
  drainPending: () => Effect.Effect<E[]>
  prependEvents: (events: E[]) => Effect.Effect<void>
}

export const EventSinkTag = <E extends BaseEvent>() =>
  Context.GenericTag<EventSinkService<E>>('EventSink')

export function makeEventSinkLayer<E extends BaseEvent>(): Layer.Layer<EventSinkService<E>> {
  const Tag = EventSinkTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const pendingRef = yield* Ref.make<E[]>([])

    return {
      append: (event: E) =>
        Ref.update(pendingRef, events => [...events, event]),

      readPending: () =>
        Ref.get(pendingRef),

      drainPending: () =>
        Ref.getAndSet(pendingRef, []),

      prependEvents: (events: E[]) =>
        Ref.update(pendingRef, pending => [...events, ...pending])
    }
  }))
}

