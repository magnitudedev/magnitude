import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import { HardwareSnapshotSchema, ModelList } from "@magnitudedev/icn/generated"
import { ModelRecipesState } from "@magnitudedev/icn/recipes"
import { defineMirroredState } from "./mirrored-state"

export const IcnHardwareMirror = defineMirroredState("GetIcnHardware", {
  stateSchema: HardwareSnapshotSchema,
  errorSchema: LocalInferenceError,
})

export const IcnInventoryMirror = defineMirroredState("GetIcnInventory", {
  stateSchema: ModelList,
  errorSchema: LocalInferenceError,
})

export const ModelRecipesMirror = defineMirroredState("GetModelRecipes", {
  stateSchema: ModelRecipesState,
  errorSchema: LocalInferenceError,
})

export const DownloadLocalModel = Rpc.make("DownloadLocalModel", {
  payload: Schema.Struct({ configurationId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const ActivateLocalModel = Rpc.make("ActivateLocalModel", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ modelId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const RestartLocalInference = Rpc.make("RestartLocalInference", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DisableLocalInference = Rpc.make("DisableLocalInference", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})
