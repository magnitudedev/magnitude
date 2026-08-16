import { Cause, Context, Duration, Effect, Layer, Option, Schedule, Schema } from "effect"
import {
  ModelsResponse,
  type CatalogModelId,
  type CatalogVariantId,
} from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"

type ModelsReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listModels"]>
>

export interface IcnModelsService
  extends IcnObservedState<ModelsResponse, ModelsReadError> {
  readonly reconcileCatalogModel: (
    modelId: CatalogModelId,
    variantId: CatalogVariantId,
  ) => ReturnType<IcnClientService["models"]["reconcileCatalogModel"]>
}

export class IcnModels extends Context.Tag("@magnitudedev/icn/IcnModels")<
  IcnModels,
  IcnModelsService
>() {}

export interface IcnModelsOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnModels = (
  options: IcnModelsOptions = {},
): Layer.Layer<IcnModels, ModelsReadError, IcnClient> =>
  Layer.scoped(
    IcnModels,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const observed = yield* makeIcnObservedState(
        {
          revision: 0,
          reconciliationComplete: false,
          catalogModels: [],
          uncataloguedPackages: [],
          diagnostics: [],
        },
        client.models.listModels({}),
        Schema.equivalence(ModelsResponse),
      )
      yield* observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh ICN models").pipe(
          Effect.annotateLogs({
            cause: Cause.pretty(Cause.fail(error)),
            detail: "cause" in error && Option.isSome(error.cause)
              ? String(error.cause.value)
              : undefined,
          }),
        )),
        Effect.option,
        Effect.repeat(Schedule.spaced(options.refreshInterval ?? "5 seconds")),
        Effect.forkScoped,
      )
      return IcnModels.of({
        ...observed,
        reconcileCatalogModel: (modelId, variantId) =>
          client.models.reconcileCatalogModel({ payload: { modelId, variantId } }).pipe(
            Effect.tap(() => observed.refresh),
          ),
      })
    }),
  )
