import { Context, Duration, Effect, Layer } from "effect"
import { AcnRpcDemand } from "@magnitudedev/protocol"
import { AcnShutdown } from "./acn-shutdown"
import {
  makeResourceUseGate,
  type ResourceUseGate,
  type ResourceUseGateSnapshot,
  type ResourceRetired,
} from "./resource-use-gate"

export interface AcnActivityTrackerApi {
  readonly withUse: <A, E, R>(
    label: string,
    effect: Effect.Effect<A, E, R>,
  ) => Effect.Effect<A, E | ResourceRetired, R>
  readonly acquire: (label: string) => Effect.Effect<Effect.Effect<void>, ResourceRetired>
  /** Ends the bootstrap lease; idempotent. The idle allowance starts here. */
  readonly ready: Effect.Effect<void>
  readonly gate: ResourceUseGate
  readonly current: Effect.Effect<ResourceUseGateSnapshot>
}
export type AcnActivityState = ResourceUseGateSnapshot

/** ACN-root demand authority. Observation never touches this service. */
export class AcnActivityTracker extends Context.Tag("AcnActivityTracker")<
  AcnActivityTracker,
  AcnActivityTrackerApi
>() {}

export const AcnActivityTrackerLive = (
  idleTimeout: Duration.DurationInput = "30 minutes",
  initiallyReady = true,
): Layer.Layer<AcnActivityTracker, never, AcnShutdown> =>
  Layer.scoped(
    AcnActivityTracker,
    Effect.gen(function* () {
      const shutdown = yield* AcnShutdown
      const gate = yield* makeResourceUseGate({
        resource: "acn",
        generation: 1,
        idleTimeout,
        retire: () =>
          shutdown.request({ reason: "idle" }).pipe(
            Effect.tap(() => Effect.logInfo("ACN demand idle deadline reached; shutting down")),
            Effect.as(true),
          ),
      })
      const releaseBootstrap = initiallyReady
        ? Effect.void
        : yield* gate.acquire("acn-startup").pipe(Effect.orDie)
      return {
        gate,
        acquire: gate.acquire,
        withUse: gate.withUse,
        ready: releaseBootstrap,
        current: gate.snapshot,
      }
    }),
  )

const withDemand = <A, E, R>(
  activity: AcnActivityTrackerApi,
  tag: string,
  next: Effect.Effect<A, E, R>,
): Effect.Effect<A, E, R> =>
  activity
    .withUse(`rpc:${tag}`, next)
    .pipe(Effect.catchTag("ResourceRetired", () => Effect.interrupt))

export const AcnRpcDemandLive: Layer.Layer<AcnRpcDemand, never, AcnActivityTracker> = Layer.effect(
  AcnRpcDemand,
  Effect.map(
    AcnActivityTracker,
    (activity) =>
      ({ rpc, next }) =>
        withDemand(activity, rpc._tag, next),
  ),
)
