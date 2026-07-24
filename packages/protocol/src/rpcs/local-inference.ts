import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import {
  CatalogCandidateIdSchema,
  LocalInferenceHardwareSchema,
  LocalModelsStateSchema,
  ModelOfferingTargetIdSchema,
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

export const DownloadCatalogModel = Rpc.make("DownloadCatalogModel", {
  payload: Schema.Struct({ id: CatalogCandidateIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const CancelCatalogModelDownload = Rpc.make("CancelCatalogModelDownload", {
  payload: Schema.Struct({ id: CatalogCandidateIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteCatalogModel = Rpc.make("DeleteCatalogModel", {
  payload: Schema.Struct({ id: CatalogCandidateIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const SelectCatalogModel = Rpc.make("SelectCatalogModel", {
  payload: Schema.Struct({ id: CatalogCandidateIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const RetryModelDownload = Rpc.make("RetryModelDownload", {
  payload: Schema.Struct({ modelId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const CancelModelDownload = Rpc.make("CancelModelDownload", {
  payload: Schema.Struct({ modelId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DismissModelDownloadFailure = Rpc.make("DismissModelDownloadFailure", {
  payload: Schema.Struct({ modelId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ modelId: ModelOfferingTargetIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const LoadModel = Rpc.make("LoadModel", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const UnloadModel = Rpc.make("UnloadModel", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})
