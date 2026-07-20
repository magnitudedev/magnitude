import { Context, Effect, Layer, PubSub, Stream } from "effect"

export interface LocalModelInventoryChangesApi {
  readonly publish: Effect.Effect<void>
  readonly stream: Stream.Stream<void>
}

/** Narrow invalidation bus for consumers of ICN model/runtime inventory. */
export class LocalModelInventoryChanges extends Context.Tag("LocalModelInventoryChanges")<
  LocalModelInventoryChanges,
  LocalModelInventoryChangesApi
>() {}

export const LocalModelInventoryChangesLive = Layer.effect(
  LocalModelInventoryChanges,
  Effect.gen(function* () {
    const pubsub = yield* PubSub.sliding<void>(1)
    return LocalModelInventoryChanges.of({
      publish: PubSub.publish(pubsub, undefined).pipe(Effect.asVoid),
      stream: Stream.fromPubSub(pubsub),
    })
  }),
)
