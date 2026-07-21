import { Context, Effect, Layer, PubSub, Ref, Stream } from "effect"

export interface LocalModelInventoryChangesApi {
  readonly publish: Effect.Effect<void>
  readonly revision: Effect.Effect<number>
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
    const revision = yield* Ref.make(0)
    return LocalModelInventoryChanges.of({
      publish: Ref.update(revision, (value) => value + 1).pipe(
        Effect.zipRight(PubSub.publish(pubsub, undefined)),
        Effect.asVoid,
      ),
      revision: Ref.get(revision),
      stream: Stream.fromPubSub(pubsub),
    })
  }),
)
