import { Context, Effect, Layer, Schema } from "effect"
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

export const makeIcnInventory = (): Layer.Layer<IcnInventory, InventoryReadError, IcnClient> =>
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
      return IcnInventory.of(observed)
    }),
  )
