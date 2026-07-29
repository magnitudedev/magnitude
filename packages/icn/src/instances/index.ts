import {
  Cause,
  Context,
  Duration,
  Effect,
  Layer,
  Schedule,
  Stream,
  SubscriptionRef,
} from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import type { ModelInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"

type InstancesReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["getModelInstances"]>
>
type InstancesWatchError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["watchModelInstances"]>
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
): Layer.Layer<
  IcnInstances,
  InstancesReadError | InstancesWatchError,
  IcnClient
> => Layer.scoped(
  IcnInstances,
  Effect.gen(function* () {
    const client = yield* IcnClient
    const retryInterval = options.retryInterval ?? "1 second"

    // Admit the invalidation watch before fetching the snapshot. A transition in
    // between is therefore either in the snapshot or invalidates it afterward.
    const initialWatch = yield* client.models.watchModelInstances({})
    const initial = yield* client.models.getModelInstances({})
    const current = yield* SubscriptionRef.make(initial)
    const refreshLock = yield* Effect.makeSemaphore(1)
    const refresh = refreshLock.withPermits(1)(
      client.models.getModelInstances({}).pipe(
        Effect.flatMap((next) => SubscriptionRef.update(current, (previous) =>
          next.revision > previous.revision ? next : previous)),
      ),
    )

    const logRefreshFailure = (error: InstancesReadError) =>
      Effect.logWarning("Unable to refresh ICN model instances").pipe(
        Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
      )
    const refreshUntilSuccessful = refresh.pipe(
      Effect.tapError(logRefreshFailure),
      Effect.retry(Schedule.spaced(retryInterval)),
    )
    const consume = (
      watch: typeof initialWatch,
    ) => watch.events.pipe(
      Stream.runForEach((invalidation) =>
        SubscriptionRef.get(current).pipe(
          Effect.flatMap((snapshot) =>
            invalidation.revision > snapshot.revision
              ? refreshUntilSuccessful
              : Effect.void),
        )),
    )
    const admitWatch = client.models.watchModelInstances({}).pipe(
      Effect.tapError((error) =>
        Effect.logWarning("Unable to reconnect ICN model instances watch").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
      Effect.retry(Schedule.spaced(retryInterval)),
    )

    // The generated stream reconnects ordinary transport interruptions with Last-Event-ID.
    // If it still terminates (for example, a reconnect admission receives a remote error),
    // re-admit the watch and refresh after admission so no transition in the gap is lost.
    yield* Effect.iterate(initialWatch, {
      while: () => true,
      body: (watch) => consume(watch).pipe(
        Effect.catchAll((error) =>
          Effect.logWarning("ICN model instances watch terminated").pipe(
            Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
          )),
        Effect.zipRight(Effect.sleep(retryInterval)),
        Effect.zipRight(admitWatch),
        Effect.tap(() => refreshUntilSuccessful),
      ),
    }).pipe(Effect.forkScoped)

    return IcnInstances.of({
      get: SubscriptionRef.get(current),
      changes: current.changes,
      initialized: Effect.succeed(true),
      refresh,
    })
  }),
)

export const IcnInstancesLive = makeIcnInstances()
