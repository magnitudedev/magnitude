import { Effect, Ref, Stream } from "effect"

/**
 * The engine's storage port, for one engine instance's log. It is storage-agnostic: the store
 * encodes what it holds.
 *
 * - `commit` appends one acceptance's events atomically, all or none. The engine commits one
 *   acceptance at a time.
 * - `read` returns the log's events in commit order.
 */
export interface EventStore<Event, E> {
  readonly read: Stream.Stream<Event, E>
  readonly commit: (events: readonly Event[]) => Effect.Effect<void, E>
}

export interface MemoryEventStore<Event> extends EventStore<Event, never> {
  readonly events: Effect.Effect<readonly Event[]>
}

export const EventStore = {
  /** A store held in memory, which also exposes what it holds. */
  memory: <Event>(): Effect.Effect<MemoryEventStore<Event>> =>
    Effect.map(Ref.make<readonly Event[]>([]), (held) => ({
      read: Stream.unwrap(Effect.map(Ref.get(held), Stream.fromIterable)),
      commit: (events) => Ref.update(held, (current) => [...current, ...events]),
      events: Ref.get(held),
    })),
}
