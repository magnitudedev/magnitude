import { Context, Effect, PubSub, Stream } from "effect"
import type { ClientInvalidation } from "@magnitudedev/sdk"

export type ClientInvalidationEvent =
  | ClientInvalidation
  | { readonly _tag: "Connected" }

/**
 * Connection-local broadcast of cache-neutral server invalidations.
 * Each domain consumes this stream and retains ownership of its own cache.
 */
export interface ClientInvalidations {
  readonly events: Stream.Stream<ClientInvalidationEvent>
  readonly mirrors: (mirrorId: string) => Stream.Stream<void>
  readonly publish: (event: ClientInvalidationEvent) => Effect.Effect<void>
}

export const ClientInvalidations = Context.GenericTag<ClientInvalidations>(
  "client/ClientInvalidations",
)

export const makeClientInvalidations = Effect.gen(function* () {
  const events = yield* PubSub.unbounded<ClientInvalidationEvent>()
  const stream = Stream.fromPubSub(events)
  return ClientInvalidations.of({
    events: stream,
    mirrors: (mirrorId) => stream.pipe(
      Stream.filter((event) => event._tag === "Connected"
        || (event._tag === "MirroredState" && event.invalidation.id === mirrorId)),
      Stream.map(() => undefined),
    ),
    publish: (event) => PubSub.publish(events, event).pipe(Effect.asVoid),
  })
})
