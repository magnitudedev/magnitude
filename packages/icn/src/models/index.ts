import { Context, Duration, Effect, Layer, Schedule, Schema } from "effect"
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
      const retryInterval = options.retryInterval ?? "1 second"
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelsResponse),
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
        new Set(["models"]),
        observed.refresh,
        "ICN models",
        retryInterval,
      ).pipe(
        Effect.forkScoped,
      )
      return IcnModels.of(observed)
    }),
  )
