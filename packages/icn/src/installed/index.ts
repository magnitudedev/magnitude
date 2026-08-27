import { Context, Duration, Effect, Layer, Schedule, Schema } from "effect"
import { InstalledModelPackagesResponse } from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type InstalledPackagesReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listInstalledModels"]>
>

export interface IcnInstalledModelsService
  extends IcnObservedState<InstalledModelPackagesResponse, InstalledPackagesReadError> {}

export class IcnInstalledModels extends Context.Tag("@magnitudedev/icn/IcnInstalledModels")<
  IcnInstalledModels,
  IcnInstalledModelsService
>() {}

export interface IcnInstalledModelsOptions {
  readonly retryInterval?: Duration.DurationInput
}

export const makeIcnInstalledModels = (
  options: IcnInstalledModelsOptions = {},
): Layer.Layer<IcnInstalledModels, InstalledPackagesReadError, IcnClient | IcnEvents> =>
  Layer.scoped(
    IcnInstalledModels,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const events = yield* IcnEvents
      const invalidations = yield* events.subscribe
      const read = client.models.listInstalledModels({})
      const retryInterval = options.retryInterval ?? "1 second"
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(InstalledModelPackagesResponse),
        { initiallyInitialized: true },
      )
      if (!initial.reconciliationComplete) {
        yield* Effect.iterate(false, {
          while: (complete) => !complete,
          body: () => Effect.sleep(retryInterval).pipe(
            Effect.zipRight(observed.refresh.pipe(Effect.retry(Schedule.spaced(retryInterval)))),
            Effect.zipRight(observed.get),
            Effect.map(({ state }) => state.reconciliationComplete),
          ),
        }).pipe(Effect.forkScoped)
      }
      yield* refreshOnIcnEvents(
        invalidations,
        new Set(["packages"]),
        observed.refresh,
        "installed model packages",
        retryInterval,
      ).pipe(
        Effect.forkScoped,
      )
      return IcnInstalledModels.of(observed)
    }),
  )
