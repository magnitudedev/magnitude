/**
 * WorkerBus
 *
 * Public API for Workers.
 * Enforces architecture by ONLY exposing publish and subscribe.
 * Workers cannot register synchronous projection handlers.
 *
 * Generic over E - the application's event union type.
 */

import { Effect, Context, Layer, Stream, Scope } from 'effect'
import { EventBusCoreTag, type BaseEvent, type EventBusCoreService } from './event-bus-core'

export interface WorkerBusService<E extends BaseEvent> {
  publish: (event: E) => Effect.Effect<void>
  subscribeToTypes: <T extends E['type']>(types: readonly T[]) => Stream.Stream<Extract<E, { type: T }>>
  stream: Stream.Stream<E>
  /**
   * Eagerly subscribe to the event bus. Returns an Effect that creates the
   * subscription immediately (buffering starts now) and yields a Stream.
   * The subscription is scoped — cleaned up when the scope closes.
   */
  subscribe: () => Effect.Effect<Stream.Stream<E>, never, Scope.Scope>
}

// Create a tag for a specific event type E
export const WorkerBusTag = <E extends BaseEvent>() =>
  Context.GenericTag<WorkerBusService<E>>('WorkerBus')

export function makeWorkerBusLayer<E extends BaseEvent>(): Layer.Layer<
  WorkerBusService<E>,
  never,
  EventBusCoreService<E>
> {
  const Tag = WorkerBusTag<E>()
  const CoreTag = EventBusCoreTag<E>()

  return Layer.scoped(Tag, Effect.gen(function* () {
    const core = yield* CoreTag

    return {
      publish: (event: E) => core.publish(event),
      subscribeToTypes: <T extends E['type']>(types: readonly T[]) => core.subscribeToTypes(types),
      stream: core.stream,
      subscribe: () => core.subscribe()
    }
  }))
}
