import { IcnClient } from "@magnitudedev/icn"
import { Context, Data, Duration, Effect, Layer, Schedule } from "effect"

export class ModelResidencyPolicyUnavailable extends Data.TaggedError("ModelResidencyPolicyUnavailable")<{
  readonly operation: "connect" | "disconnect"
  readonly message: string
}> {}

class ResidencyPolicySuperseded extends Data.TaggedError("ResidencyPolicySuperseded")<{
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
    const mutationLock = yield* Effect.makeSemaphore(1)

    const setConnected: ModelResidencyPolicy["setConnected"] = (connected) =>
      mutationLock.withPermits(1)(
        Effect.gen(function* () {
          const idleTimeout = connected ? CONNECTED_IDLE_TIMEOUT : DISCONNECTED_IDLE_TIMEOUT
          const update = Effect.gen(function* () {
            const current = yield* client.models.getModelResidencyPolicy({})
            const nextGeneration = current.generation + 1
            yield* client.models.setModelResidencyPolicy({
              payload: {
                generation: nextGeneration,
                idleTimeoutSeconds: Duration.toSeconds(idleTimeout),
              },
            })
            const observed = yield* client.models.getModelResidencyPolicy({})
            if (observed.generation < nextGeneration
              || observed.idleTimeoutSeconds !== Duration.toSeconds(idleTimeout)) {
              return yield* new ResidencyPolicySuperseded({
                message: "residency policy was superseded before acknowledgement",
              })
            }
          })
          yield* update.pipe(
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
        })
      )

    return { setConnected }
  })
)
