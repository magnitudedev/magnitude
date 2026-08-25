import { Option, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { ProviderIdSchema, ProviderModelIdSchema } from "@magnitudedev/ai/provider/model"
import {
  LocalInferenceError,
  ModelPreferenceMutationFailed,
  ModelSlotUpdateError,
} from "../errors"
import {
  CatalogModelReconciliationAdmissionSchema,
  LocalInferenceHardwareSchema,
  ModelCatalogStateSchema,
  ModelDownloadIdSchema,
  ModelInstanceIdSchema,
  ModelInstancesStateSchema,
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
const GetInstances = stateQuery("GetModelInstances", ModelInstancesStateSchema)
const GetLocalEnvironment = stateQuery("GetLocalInferenceEnvironment", LocalInferenceHardwareSchema)

const RefreshCatalog = Mutation.make("RefreshModelCatalog", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    providerId: Schema.optionalWith(ProviderIdSchema, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: Schema.Never,
})

const slotScope = (slotId: SlotId) => Mutation.MutationScope(`model-slot:${slotId}`)
const slotMutationScope = (slotId: SlotId) => slotId === "primary"
  ? turnAdmissionScope
  : slotScope(slotId)
const invalidateCatalog = QueryClient.invalidate(GetCatalog.match())
const invalidateSlots = QueryClient.invalidate(GetSlots.match())

const AssignSlot = Mutation.make("AssignModelSlot", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ slotId: SlotIdSchema, selection: SlotSelectionSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotMutationScope(slotId),
  synchronize: () => invalidateSlots,
})

const ClearSlot = Mutation.make("ClearModelSlot", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotMutationScope(slotId),
  synchronize: () => invalidateSlots,
})

const SetFavorite = Mutation.make("SetModelFavorite", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ model: ProviderModelIdentitySchema, favorite: Schema.Boolean }),
  success: Schema.Struct({}),
  error: ModelPreferenceMutationFailed,
  scope: ({ model }) => Mutation.MutationScope(`model-favorite:${model.providerId}:${model.providerModelId}`),
  synchronize: () => invalidateSlots,
})

const InstallLocalModel = Mutation.make("InstallLocalModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: CatalogModelReconciliationAdmissionSchema,
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => invalidateCatalog,
})

const CancelModelDownload = Mutation.make("CancelModelDownload", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ downloadId }) => Mutation.MutationScope(`model-download:${downloadId}`),
  synchronize: () => invalidateCatalog,
})

const AcknowledgeModelDownloadFailure = Mutation.make("AcknowledgeModelDownloadFailure", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ downloadId }) => Mutation.MutationScope(`model-download:${downloadId}`),
  synchronize: () => invalidateCatalog,
})

const UninstallLocalModel = Mutation.make("UninstallLocalModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ modelId: ProviderModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
  synchronize: () => invalidateCatalog,
})

const LoadSlot = Mutation.make("LoadModelSlot", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({ instanceId: ModelInstanceIdSchema }),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => invalidateSlots,
})

const StopSlot = Mutation.make("StopModelSlot", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({ instanceId: ModelInstanceIdSchema }),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => invalidateSlots,
})

const PreviewSlotLoad = Query.make("PreviewModelSlotLoad", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: ModelLoadPlanSchema,
  error: LocalInferenceError,
})

export const Models = Group.make({
  GetCatalog,
  GetSlots,
  GetInstances,
  GetLocalEnvironment,
  PreviewSlotLoad,
  RefreshCatalog,
  AssignSlot,
  ClearSlot,
  SetFavorite,
  InstallLocalModel,
  CancelModelDownload,
  AcknowledgeModelDownloadFailure,
  UninstallLocalModel,
  LoadSlot,
  StopSlot,
})
