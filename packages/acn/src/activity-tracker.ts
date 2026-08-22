import { Context, Effect, Layer, Option } from "effect"
import { AcnRpcDemand, AcnRpcDemandPolicyTag } from "@magnitudedev/acn-protocol"
import {
  type ResourceUseGateSnapshot,
  type ResourceRetired,
} from "./resource-use-gate"
import { AcnServiceLifecycle } from "./service-lifecycle"

export interface AcnActivityTrackerApi {
  readonly withUse: <A, E, R>(
    label: string,
    effect: Effect.Effect<A, E, R>,
  ) => Effect.Effect<A, E | ResourceRetired, R>
  readonly acquire: (label: string) => Effect.Effect<Effect.Effect<void>, ResourceRetired>
  readonly joinIfBusy: (
    label: string,
  ) => Effect.Effect<Option.Option<Effect.Effect<void>>, ResourceRetired>
  readonly current: Effect.Effect<ResourceUseGateSnapshot>
}
export type AcnActivityState = ResourceUseGateSnapshot

/** ACN-root demand authority. Observation never touches this service. */
export class AcnActivityTracker extends Context.Tag("AcnActivityTracker")<
  AcnActivityTracker,
  AcnActivityTrackerApi
>() {}

export const AcnActivityTrackerLive: Layer.Layer<
  AcnActivityTracker,
  never,
  AcnServiceLifecycle
> =
  Layer.effect(
    AcnActivityTracker,
    Effect.gen(function* () {
      const lifecycle = yield* AcnServiceLifecycle
      return {
        acquire: lifecycle.acquireActivity,
        joinIfBusy: lifecycle.joinActivityIfBusy,
        withUse: lifecycle.withActivity,
        current: lifecycle.activity,
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
      ({ rpc, next }) => {
        const demand = Context.getOption(rpc.annotations, AcnRpcDemandPolicyTag)
        return Option.isSome(demand) && demand.value
          ? withDemand(activity, rpc._tag, next)
          : next
      },
  ),
)
