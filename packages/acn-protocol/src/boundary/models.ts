import { Option, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { ProviderIdSchema, ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"
import {
  LocalInferenceError,
  ModelPreferenceMutationFailed,
  ModelSlotUpdateError,
} from "../errors"
import {
  LocalInferenceHardwareSchema,
  ModelCatalogStateSchema,
  ModelLoadPlanSchema,
  ModelSlotsStateSchema,
  ProviderModelIdentitySchema,
  SlotIdSchema,
  SlotSelectionSchema,
  type SlotId,
} from "../schemas/model-state"
import { turnAdmissionScope } from "./configuration"

const stateQuery = <const Name extends string, A, I>(name: Name, schema: Schema.Schema<A, I>) => Query.make(name, {
  payload: Schema.Struct({}),
  success: schema,
  error: Schema.Never,
  staleTime: Infinity,
  gcTime: Infinity,
})

const GetCatalog = stateQuery("GetModelCatalog", ModelCatalogStateSchema)
const GetSlots = stateQuery("GetModelSlots", ModelSlotsStateSchema)
const GetLocalEnvironment = stateQuery("GetLocalInferenceEnvironment", LocalInferenceHardwareSchema)

const RefreshCatalog = Mutation.make("RefreshModelCatalog", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    providerId: Schema.optionalWith(ProviderIdSchema, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: Schema.Never,
  synchronize: () => QueryClient.refetch(GetCatalog.match()),
})

const slotScope = (slotId: SlotId) => Mutation.MutationScope(`model-slot:${slotId}`)
const slotMutationScope = (slotId: SlotId) => slotId === "primary"
  ? turnAdmissionScope
  : slotScope(slotId)
const synchronizeCatalog = QueryClient.refetch(GetCatalog.match())
const synchronizeSlots = QueryClient.refetch(GetSlots.match())

const AssignSlot = Mutation.make("AssignModelSlot", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ slotId: SlotIdSchema, selection: SlotSelectionSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotMutationScope(slotId),
  synchronize: () => synchronizeSlots,
})

const ClearSlot = Mutation.make("ClearModelSlot", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotMutationScope(slotId),
  synchronize: () => synchronizeSlots,
})

const SetFavorite = Mutation.make("SetModelFavorite", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ model: ProviderModelIdentitySchema, favorite: Schema.Boolean }),
  success: Schema.Struct({}),
  error: ModelPreferenceMutationFailed,
  scope: ({ model }) => Mutation.MutationScope(`model-favorite:${model.providerId}:${model.providerModelId}`),
  synchronize: () => synchronizeSlots,
})

const SyncLocalModel = Mutation.make("SyncLocalModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => synchronizeCatalog,
})

const CancelLocalModelSync = Mutation.make("CancelLocalModelSync", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => synchronizeCatalog,
})

const AcknowledgeLocalModelSyncFailure = Mutation.make("AcknowledgeLocalModelSyncFailure", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => synchronizeCatalog,
})

const RemoveLocalModel = Mutation.make("RemoveLocalModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => synchronizeCatalog,
})

const LoadSlot = Mutation.make("LoadModelSlot", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => synchronizeSlots,
})

const StopSlot = Mutation.make("StopModelSlot", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => synchronizeSlots,
})

const PreviewSlotLoad = Query.make("PreviewModelSlotLoad", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: ModelLoadPlanSchema,
  error: LocalInferenceError,
})

export const Models = Group.make({
  GetCatalog,
  GetSlots,
  GetLocalEnvironment,
  PreviewSlotLoad,
  RefreshCatalog,
  AssignSlot,
  ClearSlot,
  SetFavorite,
  SyncLocalModel,
  CancelLocalModelSync,
  AcknowledgeLocalModelSyncFailure,
  RemoveLocalModel,
  LoadSlot,
  StopSlot,
})
