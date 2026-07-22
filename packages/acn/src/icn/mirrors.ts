import { Context, Effect, Layer } from "effect"
import {
  IcnHardwareMirror,
  IcnInventoryMirror,
  ModelRecipesMirror,
  type MirroredSnapshot,
} from "@magnitudedev/protocol"
import {
  IcnHardware,
  IcnInventory,
  IcnRecipes,
  type Generated,
  type ModelRecipesState,
} from "@magnitudedev/icn"
import { bindMirroredState, MirroredStateChanges } from "../mirrored-state"

export interface IcnMirrorsService {
  readonly hardware: Effect.Effect<MirroredSnapshot<Generated.HardwareSnapshotSchema>>
  readonly inventory: Effect.Effect<MirroredSnapshot<Generated.ModelList>>
  readonly recipes: Effect.Effect<MirroredSnapshot<ModelRecipesState>>
}

export class IcnMirrors extends Context.Tag("Acn/IcnMirrors")<
  IcnMirrors,
  IcnMirrorsService
>() {}

export const makeIcnMirrors = (): Layer.Layer<
  IcnMirrors,
  never,
  IcnHardware | IcnInventory | IcnRecipes | MirroredStateChanges
> => Layer.scoped(
  IcnMirrors,
  Effect.gen(function* () {
    const hardware = yield* IcnHardware
    const inventory = yield* IcnInventory
    const recipes = yield* IcnRecipes
    const hardwareMirror = yield* bindMirroredState(IcnHardwareMirror, hardware)
    const inventoryMirror = yield* bindMirroredState(IcnInventoryMirror, inventory)
    const recipesMirror = yield* bindMirroredState(ModelRecipesMirror, recipes)

    return IcnMirrors.of({
      hardware: hardwareMirror.get,
      inventory: inventoryMirror.get,
      recipes: recipesMirror.get,
    })
  }),
)
