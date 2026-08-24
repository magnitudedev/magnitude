import { Context, Duration, Effect, Layer, Schema } from "effect"
import { ModelsResponse } from "@magnitudedev/icn-protocol/schemas"
import { IcnClient, type IcnClientService } from "../client.js"
import { makeIcnObservedState, type IcnObservedState } from "../observed-state.js"
import { IcnEvents, refreshOnIcnEvents } from "../events/index.js"

type ModelsReadError = Effect.Effect.Error<
  ReturnType<IcnClientService["models"]["listModels"]>
>

export interface IcnModelsService
  extends IcnObservedState<ModelsResponse, ModelsReadError> {}

export class IcnModels extends Context.Tag("@magnitudedev/icn/IcnModels")<
  IcnModels,
  IcnModelsService
>() {}

export interface IcnModelsOptions {
  readonly retryInterval?: Duration.DurationInput
}

export const makeIcnModels = (
  options: IcnModelsOptions = {},
): Layer.Layer<IcnModels, ModelsReadError, IcnClient | IcnEvents> =>
  Layer.scoped(
    IcnModels,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const events = yield* IcnEvents
      const invalidations = yield* events.subscribe
      const read = client.models.listModels({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelsResponse),
        { initiallyInitialized: true },
      )
      yield* refreshOnIcnEvents(
        invalidations,
        new Set(["models"]),
        observed.refresh,
        "ICN models",
        options.retryInterval ?? "1 second",
      ).pipe(
        Effect.forkScoped,
      )
      return IcnModels.of(observed)
    }),
  )
