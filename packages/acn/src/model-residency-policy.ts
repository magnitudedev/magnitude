import { IcnClient } from "@magnitudedev/icn"
import { Context, Data, Duration, Effect, Layer, Ref, Schedule } from "effect"

export class ModelResidencyPolicyUnavailable extends Data.TaggedError("ModelResidencyPolicyUnavailable")<{
  readonly operation: "connect" | "disconnect"
  readonly message: string
}> {}

export interface ModelResidencyPolicy {
  readonly setConnected: (connected: boolean) => Effect.Effect<void, ModelResidencyPolicyUnavailable>
}

export const ModelResidencyPolicy = Context.GenericTag<ModelResidencyPolicy>("ModelResidencyPolicy")

const CONNECTED_IDLE_TIMEOUT = Duration.minutes(60)
const DISCONNECTED_IDLE_TIMEOUT = Duration.minutes(10)

export const ModelResidencyPolicyLive: Layer.Layer<ModelResidencyPolicy, never, IcnClient> = Layer.effect(
  ModelResidencyPolicy,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const generation = yield* Ref.make(0)
    const mutationLock = yield* Effect.makeSemaphore(1)

    const setConnected: ModelResidencyPolicy["setConnected"] = (connected) =>
      mutationLock.withPermits(1)(
        Effect.gen(function* () {
          const currentGeneration = yield* Ref.get(generation)
          const nextGeneration = currentGeneration + 1
          const idleTimeout = connected ? CONNECTED_IDLE_TIMEOUT : DISCONNECTED_IDLE_TIMEOUT
          yield* client.models
            .setModelResidencyPolicy({
              payload: {
                generation: nextGeneration,
                idleTimeoutSeconds: Duration.toSeconds(idleTimeout),
              },
            })
            .pipe(
              Effect.retry({ schedule: Schedule.spaced(Duration.millis(100)) }),
              Effect.timeout(Duration.seconds(2)),
              Effect.mapError(
                (error) =>
                  new ModelResidencyPolicyUnavailable({
                    operation: connected ? "connect" : "disconnect",
                    message: String(error),
                  })
              )
            )
          yield* Ref.set(generation, nextGeneration)
        })
      )

    return { setConnected }
  })
)
