import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import {
  ModelDownloadIdSchema,
  LocalInferenceHardwareSchema,
  LocalModelsStateSchema,
  LocalModelInstallationAdmissionSchema,
  ModelInstanceIdSchema,
  ModelLoadAdmissionSchema,
  ModelLoadPlanSchema,
  ModelServingConfigurationIdSchema,
  SlotIdSchema,
  SlotSelectionSchema,
} from "../schemas/model-state"
import { defineMirroredState } from "./mirrored-state"

export const LocalInferenceHardwareMirror = defineMirroredState("GetLocalInferenceHardware", {
  stateSchema: LocalInferenceHardwareSchema,
  errorSchema: Schema.Never,
})

export const LocalModelsMirror = defineMirroredState("GetLocalModels", {
  stateSchema: LocalModelsStateSchema,
  errorSchema: Schema.Never,
})

export const InstallModel = Rpc.make("InstallModel", {
  payload: Schema.Struct({ configurationId: ModelServingConfigurationIdSchema }),
  success: LocalModelInstallationAdmissionSchema,
  error: LocalInferenceError,
})

export const CancelModelDownload = Rpc.make("CancelModelDownload", {
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DismissModelDownloadFailure = Rpc.make("DismissModelDownloadFailure", {
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ configurationId: ModelServingConfigurationIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const LoadModel = Rpc.make("LoadModel", {
  payload: Schema.Struct({ slotId: SlotIdSchema, selection: SlotSelectionSchema }),
  success: ModelLoadAdmissionSchema,
  error: LocalInferenceError,
})

export const PreviewModelLoad = Rpc.make("PreviewModelLoad", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: ModelLoadPlanSchema,
  error: LocalInferenceError,
})

export const StopModel = Rpc.make("StopModel", {
  payload: Schema.Struct({ instanceId: ModelInstanceIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})
