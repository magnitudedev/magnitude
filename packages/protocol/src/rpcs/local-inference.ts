import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import { LocalInferenceState } from "../schemas/local-inference"
import { defineMirroredState } from "./mirrored-state"

export const LocalInferenceMirror = defineMirroredState("GetLocalInferenceState", {
  stateSchema: LocalInferenceState,
  errorSchema: LocalInferenceError,
})

export const DownloadLocalModel = Rpc.make("DownloadLocalModel", {
  payload: Schema.Struct({ configurationId: Schema.String, requestId: Schema.String }),
  success: Schema.Struct({ operationId: Schema.String }),
  error: LocalInferenceError,
})

export const ActivateLocalModel = Rpc.make("ActivateLocalModel", {
  payload: Schema.Struct({ selectionId: Schema.String, requestId: Schema.String }),
  success: Schema.Struct({ operationId: Schema.String }),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ selectionId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const RestartLocalInference = Rpc.make("RestartLocalInference", {
  payload: Schema.Struct({ requestId: Schema.String }),
  success: Schema.Struct({ operationId: Schema.String }),
  error: LocalInferenceError,
})

export const DisableLocalInference = Rpc.make("DisableLocalInference", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})
