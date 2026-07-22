import { Cause, Context, Duration, Effect, Layer, Schema } from "effect"
import { IcnClient, type IcnClientService } from "../client.js"
import { ModelList } from "../generated/schemas.js"
import {
  makeIcnObservedState,
  type IcnObservedState,
} from "../observed-state.js"

type InventoryReadError = Effect.Effect.Error<ReturnType<IcnClientService["models"]["listModels"]>>

export interface IcnInventoryService extends IcnObservedState<ModelList, InventoryReadError> {}

export class IcnInventory extends Context.Tag("@magnitudedev/icn/IcnInventory")<
  IcnInventory,
  IcnInventoryService
>() {}

export interface IcnInventoryOptions {
  readonly refreshInterval?: Duration.DurationInput
}

export const makeIcnInventory = (
  options: IcnInventoryOptions = {},
): Layer.Layer<IcnInventory, InventoryReadError, IcnClient> =>
  Layer.scoped(
    IcnInventory,
    Effect.gen(function* () {
      const client = yield* IcnClient
      const read = client.models.listModels({})
      const initial = yield* read
      const observed = yield* makeIcnObservedState(
        initial,
        read,
        Schema.equivalence(ModelList),
      )
      const refreshObservedState = observed.refresh.pipe(
        Effect.tapError((error) => Effect.logWarning("Unable to refresh ICN model inventory").pipe(
          Effect.annotateLogs({ cause: Cause.pretty(Cause.fail(error)) }),
        )),
        Effect.option,
        Effect.asVoid,
      )

      yield* refreshObservedState.pipe(
        Effect.delay(options.refreshInterval ?? "150 millis"),
        Effect.forever,
        Effect.forkScoped,
      )

      return IcnInventory.of(observed)
    }),
  )
