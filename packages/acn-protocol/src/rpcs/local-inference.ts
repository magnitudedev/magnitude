import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"
import { LocalInferenceError } from "../errors"
import {
  DownloadAttemptIdSchema,
  LocalInferenceHardwareSchema,
  LocalModelsStateSchema,
  ModelDownloadAdmissionSchema,
  ModelInstanceIdSchema,
  ModelLoadAdmissionSchema,
  ModelLoadPlanSchema,
  ModelOfferingTargetIdSchema,
  ModelServingConfigurationIdSchema,
  SlotIdSchema,
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

export const DownloadModel = Rpc.make("DownloadModel", {
  payload: Schema.Struct({ targetId: ModelOfferingTargetIdSchema }),
  success: ModelDownloadAdmissionSchema,
  error: LocalInferenceError,
})

export const CreateLocalModelOffering = Rpc.make("CreateLocalModelOffering", {
  payload: Schema.Struct({ configurationId: ModelServingConfigurationIdSchema }),
  success: ProviderModelIdSchema,
  error: LocalInferenceError,
})

export const CancelModelDownload = Rpc.make("CancelModelDownload", {
  payload: Schema.Struct({
    attemptIds: Schema.NonEmptyArray(DownloadAttemptIdSchema),
  }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DismissModelDownloadFailure = Rpc.make("DismissModelDownloadFailure", {
  payload: Schema.Struct({ targetId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ targetId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const LoadModel = Rpc.make("LoadModel", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
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
