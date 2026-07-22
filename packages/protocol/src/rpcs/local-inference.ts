import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { LocalInferenceError } from "../errors"
import {
  LocalInferenceHardwareSchema,
  LocalModelIdSchema,
  LocalModelInventoryStateSchema,
  SlotSelectionSchema,
  SlotIdSchema,
} from "../schemas/model-state"
import { defineMirroredState } from "./mirrored-state"

export const LocalInferenceHardwareMirror = defineMirroredState("GetLocalInferenceHardware", {
  stateSchema: LocalInferenceHardwareSchema,
  errorSchema: Schema.Never,
})

export const LocalModelInventoryMirror = defineMirroredState("GetLocalModelInventory", {
  stateSchema: LocalModelInventoryStateSchema,
  errorSchema: Schema.Never,
})

export const DownloadLocalModel = Rpc.make("DownloadLocalModel", {
  payload: Schema.Struct({ localModelId: LocalModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const DeleteLocalModel = Rpc.make("DeleteLocalModel", {
  payload: Schema.Struct({ localModelId: LocalModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const LoadModelSlot = Rpc.make("LoadModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema, selection: SlotSelectionSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const ReloadModelSlot = Rpc.make("ReloadModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})

export const UnloadModelSlot = Rpc.make("UnloadModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
})
