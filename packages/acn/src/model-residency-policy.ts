import { IcnClient } from "@magnitudedev/icn"
import {
  Cause,
  Context,
  Data,
  Duration,
  Effect,
  Exit,
  Layer,
  Option,
  Stream,
  SubscriptionRef,
} from "effect"

class ResidencyPolicySuperseded extends Data.TaggedError("ResidencyPolicySuperseded")<{
  readonly message: string
}> {}

export interface ModelResidencyPolicy {
  readonly setConnected: (connected: boolean) => Effect.Effect<void>
}

export const ModelResidencyPolicy = Context.GenericTag<ModelResidencyPolicy>("ModelResidencyPolicy")

const CONNECTED_IDLE_TIMEOUT = Duration.minutes(60)
const DISCONNECTED_IDLE_TIMEOUT = Duration.minutes(10)
const POLICY_ATTEMPT_TIMEOUT = Duration.seconds(2)
const POLICY_RETRY_DELAY = Duration.millis(250)

interface DesiredResidencyPolicy {
  readonly generation: number
  readonly connected: boolean
}

export const ModelResidencyPolicyLive: Layer.Layer<ModelResidencyPolicy, never, IcnClient> = Layer.scoped(
  ModelResidencyPolicy,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const desired = yield* SubscriptionRef.make(Option.none<DesiredResidencyPolicy>())

    const publish = (connected: boolean) => Effect.gen(function* () {
      const idleTimeout = connected ? CONNECTED_IDLE_TIMEOUT : DISCONNECTED_IDLE_TIMEOUT
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

    const reconcile = (target: DesiredResidencyPolicy): Effect.Effect<void> => Effect.gen(function* () {
      while (true) {
        const current = yield* SubscriptionRef.get(desired)
        if (Option.isNone(current) || current.value.generation !== target.generation) return
        const attempt = yield* publish(target.connected).pipe(
          Effect.timeout(POLICY_ATTEMPT_TIMEOUT),
          Effect.exit,
        )
        if (Exit.isSuccess(attempt)) return
        yield* Effect.logWarning("Unable to publish model residency policy; retrying").pipe(
          Effect.annotateLogs({
            connected: target.connected,
            cause: Cause.pretty(attempt.cause),
          }),
        )
        yield* Effect.sleep(POLICY_RETRY_DELAY)
      }
    })

    yield* desired.changes.pipe(
      Stream.filterMap((target) => target),
      Stream.runForEach(reconcile),
      Effect.forkScoped,
    )

    const setConnected: ModelResidencyPolicy["setConnected"] = (connected) =>
      SubscriptionRef.update(desired, (current) => {
        if (Option.exists(current, (target) => target.connected === connected)) return current
        return Option.some({
          generation: Option.match(current, {
            onNone: () => 1,
            onSome: (target) => target.generation + 1,
          }),
          connected,
        })
      })

    return { setConnected }
  })
)
