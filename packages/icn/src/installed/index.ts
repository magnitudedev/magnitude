import { Cause, Context, Duration, Effect, Layer, Schedule, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { InstalledModelPackagesResponse as InstalledModelPackagesResponseSchema } from "../generated/schemas.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"

type InstalledReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listInstalledModels"]>
>

export interface IcnInstalledModelsService
  extends IcnObservedState<InstalledModelPackagesResponseSchema, InstalledReadError> {}

export class IcnInstalledModels extends Context.Tag("@magnitudedev/icn/IcnInstalledModels")<
  IcnInstalledModels,
  IcnInstalledModelsService
>() {}

export interface IcnInstalledModelsOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnInstalledModels = (
  options: IcnInstalledModelsOptions = {},
): Layer.Layer<IcnInstalledModels, InstalledReadError, IcnClient> =>
  Layer.scoped(
    IcnInstalledModels,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.models.listInstalledModels({})
      const observed = yield* makeIcnObservedState(
        { packages: [] },
        read,
        Schema.equivalence(InstalledModelPackagesResponseSchema),
      )
      yield* observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh installed model packages").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.option,
        Effect.repeat(Schedule.spaced(options.refreshInterval ?? "5 seconds")),
        Effect.forkScoped,
      )
      return IcnInstalledModels.of(observed)
    }),
  )
