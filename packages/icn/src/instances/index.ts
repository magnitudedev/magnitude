import { Context, Duration, Effect, Layer, Stream, SubscriptionRef } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import type { ModelInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type InstancesReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["getModelInstances"]>
>

export interface IcnInstancesService {
  readonly get: Effect.Effect<ModelInstancesSnapshot>
  readonly changes: Stream.Stream<ModelInstancesSnapshot>
  readonly initialized: Effect.Effect<boolean>
  readonly refresh: Effect.Effect<void, InstancesReadError>
}

export class IcnInstances extends Context.Tag("@magnitudedev/icn/IcnInstances")<
  IcnInstances,
  IcnInstancesService
>() {}

export interface IcnInstancesOptions {
  readonly retryInterval?: Duration.DurationInput
}

export const makeIcnInstances = (
  options: IcnInstancesOptions = {},
): Layer.Layer<IcnInstances, InstancesReadError, IcnClient | IcnEvents> => Layer.scoped(
  IcnInstances,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const events = yield* IcnEvents

    // Subscribe before reading the snapshot so an invalidation racing the read
    // is buffered for this projection.
    const invalidations = yield* events.subscribe
    const initial = yield* client.models.getModelInstances({})
    const current = yield* SubscriptionRef.make(initial)
    const refreshLock = yield* Effect.makeSemaphore(1)
    const refresh = refreshLock.withPermits(1)(
      client.models.getModelInstances({}).pipe(
        Effect.flatMap((next) => SubscriptionRef.update(current, (previous) =>
          next.revision > previous.revision ? next : previous)),
      ),
    )

    yield* refreshOnIcnEvents(
      invalidations,
      new Set(["instances"]),
      refresh,
      "ICN model instances",
      options.retryInterval ?? "1 second",
    ).pipe(Effect.forkScoped)

    return IcnInstances.of({
      get: SubscriptionRef.get(current),
      changes: current.changes,
      initialized: Effect.succeed(true),
      refresh,
    })
  }),
)

export const IcnInstancesLive = makeIcnInstances()
