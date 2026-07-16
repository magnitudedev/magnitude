import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import {
  LocalInferenceState,
  LocalInferenceUsageSelection,
} from "../schemas/local-inference"
import { MirroredResourceInvalidationSchema } from "../schemas/mirrored-resource"
import { StreamHeartbeat } from "../schemas/events"

export const GetLocalInferenceState = Rpc.make("GetLocalInferenceState", {
  payload: Schema.Struct({}),
  success: LocalInferenceState,
  error: LocalInferenceError,
})

export const WatchLocalInferenceState = Rpc.make("WatchLocalInferenceState", {
  payload: Schema.Struct({}),
  success: Schema.Union(MirroredResourceInvalidationSchema, StreamHeartbeat),
  error: LocalInferenceError,
  stream: true,
})

export const ConfigureLocalInferenceUsage = Rpc.make("ConfigureLocalInferenceUsage", {
  payload: LocalInferenceUsageSelection,
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const InstallLocalInferenceDistribution = Rpc.make("InstallLocalInferenceDistribution", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DownloadLocalModel = Rpc.make("DownloadLocalModel", {
  payload: Schema.Struct({ configurationId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const ActivateLocalModel = Rpc.make("ActivateLocalModel", {
  payload: Schema.Struct({ selectionId: Schema.String }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ selectionId: Schema.String }),
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
