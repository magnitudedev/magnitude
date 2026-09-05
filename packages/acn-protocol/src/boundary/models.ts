import { Rpc } from "@effect/rpc"
import { replaySafe, atMostOnce } from "../transport/recovery"
import { Option, Schema } from "effect"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import {
  LocalInferenceError,
  ModelPreferenceMutationFailed,
  ModelSlotUpdateError,
} from "../errors"
import {
  LocalInferenceHardwareSchema,
  CatalogFormModelIdSchema,
  ModelIdSchema,
  ModelCatalogStateSchema,
  ModelLoadPlanSchema,
  ModelSlotsStateSchema,
  ProviderModelIdentitySchema,
  SlotIdSchema,
  SlotSelectionSchema,
} from "../schemas/model-state"

const GetCatalog = Rpc.make("GetModelCatalog", {
  payload: Schema.Struct({}),
  success: ModelCatalogStateSchema,
}).pipe(replaySafe)

const GetSlots = Rpc.make("GetModelSlots", {
  payload: Schema.Struct({}),
  success: ModelSlotsStateSchema,
}).pipe(replaySafe)

const GetLocalEnvironment = Rpc.make("GetLocalInferenceEnvironment", {
  payload: Schema.Struct({}),
  success: LocalInferenceHardwareSchema,
}).pipe(replaySafe)

const RefreshCatalog = Rpc.make("RefreshModelCatalog", {
  payload: Schema.Struct({
    providerId: Schema.optionalWith(ProviderIdSchema, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: Schema.Never,
}).pipe(atMostOnce)

const AssignSlot = Rpc.make("AssignModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema, selection: SlotSelectionSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
}).pipe(replaySafe)

const ClearSlot = Rpc.make("ClearModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
}).pipe(replaySafe)

const SetFavorite = Rpc.make("SetModelFavorite", {
  payload: Schema.Struct({ model: ProviderModelIdentitySchema, favorite: Schema.Boolean }),
  success: Schema.Struct({}),
  error: ModelPreferenceMutationFailed,
}).pipe(replaySafe)

const SyncLocalModel = Rpc.make("SyncLocalModel", {
  payload: Schema.Struct({ modelId: CatalogFormModelIdSchema }),
  success: Schema.Struct({ outcome: Schema.Literal("Started", "AlreadyCurrent") }),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const CancelLocalModelSync = Rpc.make("CancelLocalModelSync", {
  payload: Schema.Struct({ modelId: CatalogFormModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const AcknowledgeLocalModelSyncFailure = Rpc.make(
  "AcknowledgeLocalModelSyncFailure",
  {
    payload: Schema.Struct({ modelId: CatalogFormModelIdSchema }),
    success: Schema.Struct({}),
    error: LocalInferenceError,
  }
).pipe(atMostOnce)

const RemoveLocalModel = Rpc.make("RemoveLocalModel", {
  payload: Schema.Struct({ modelId: CatalogFormModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const LoadLocalModel = Rpc.make("LoadLocalModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const StopActiveLocalModel = Rpc.make("StopActiveLocalModel", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const LoadSlot = Rpc.make("LoadModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const StopSlot = Rpc.make("StopModelSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const PreviewSlotLoad = Rpc.make("PreviewModelSlotLoad", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: ModelLoadPlanSchema,
  error: LocalInferenceError,
}).pipe(replaySafe)

export const Models = {
  getCatalog: GetCatalog,
  getSlots: GetSlots,
  getLocalEnvironment: GetLocalEnvironment,
  refreshCatalog: RefreshCatalog,
  assignSlot: AssignSlot,
  clearSlot: ClearSlot,
  setFavorite: SetFavorite,
  syncLocalModel: SyncLocalModel,
  cancelLocalModelSync: CancelLocalModelSync,
  acknowledgeLocalModelSyncFailure: AcknowledgeLocalModelSyncFailure,
  removeLocalModel: RemoveLocalModel,
  load: LoadLocalModel,
  stop: StopActiveLocalModel,
  loadSlot: LoadSlot,
  stopSlot: StopSlot,
  previewSlotLoad: PreviewSlotLoad,
}
