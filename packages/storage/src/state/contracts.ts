import type { Effect, Stream } from 'effect'

export interface StateHandle<A, E> {
  readonly get: Effect.Effect<A>
  readonly changes: Stream.Stream<A>
  readonly modify: <B>(
    transition: (current: A) => readonly [result: B, state: A],
  ) => Effect.Effect<B, E>
  readonly update: (
    transition: (current: A) => A,
  ) => Effect.Effect<A, E>
}
