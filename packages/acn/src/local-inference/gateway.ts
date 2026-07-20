import { Context, Effect, Layer, PubSub, Stream } from "effect"

export interface LocalInferenceChangesApi {
  readonly publish: Effect.Effect<void>
  readonly stream: Stream.Stream<void>
}

export class LocalInferenceChanges extends Context.Tag("LocalInferenceChanges")<
  LocalInferenceChanges,
  LocalInferenceChangesApi
>() {}

export const LocalInferenceChangesLive = Layer.effect(
  LocalInferenceChanges,
  Effect.gen(function* () {
    const pubsub = yield* PubSub.unbounded<void>()
    return LocalInferenceChanges.of({
      publish: PubSub.publish(pubsub, undefined).pipe(Effect.asVoid),
      stream: Stream.fromPubSub(pubsub),
    })
  }),
)
