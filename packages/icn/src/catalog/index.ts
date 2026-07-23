import {
  Cause,
  Context,
  Duration,
  Effect,
  Layer,
  Schedule,
  Schema,
  Stream,
} from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { RecommendableModelCatalog as RecommendableModelCatalogSchema } from "../generated/schemas.js"
import { makeIcnObservedState, type IcnObservedSnapshot } from "../observed-state.js"

type CatalogReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["getRecommendableModelCatalog"]>
>

export interface IcnCatalogService {
  readonly get: Effect.Effect<IcnObservedSnapshot<RecommendableModelCatalogSchema>>
  readonly changes: Stream.Stream<IcnObservedSnapshot<RecommendableModelCatalogSchema>>
  readonly ready: Effect.Effect<boolean>
  readonly refresh: Effect.Effect<void, CatalogReadError>
}

export class IcnCatalog extends Context.Tag("@magnitudedev/icn/IcnCatalog")<
  IcnCatalog,
  IcnCatalogService
>() {}

export interface IcnCatalogOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnCatalog = (
  options: IcnCatalogOptions = {},
): Layer.Layer<IcnCatalog, CatalogReadError, IcnClient> =>
  Layer.scoped(
    IcnCatalog,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.models.getRecommendableModelCatalog({})
      const observed = yield* makeIcnObservedState(
        {
          models: [],
          diagnostics: [],
        },
        read,
        Schema.equivalence(RecommendableModelCatalogSchema),
      )
      yield* observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh ICN model catalog").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.option,
        Effect.repeat(Schedule.spaced(options.refreshInterval ?? "1 hour")),
        Effect.forkScoped,
      )
      return IcnCatalog.of({
        get: observed.get,
        changes: observed.changes,
        ready: observed.initialized,
        refresh: observed.refresh,
      })
    }),
  )
