import { Data, Effect, Option, Schedule, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { LocalInferenceError } from "../errors"
import { MirroredSnapshotSchema } from "../schemas/mirrored-state"
import {
  ModelDownloadIdSchema,
  CatalogIdentitySchema,
  LocalInferenceHardwareSchema,
  LocalModelsStateSchema,
  CatalogModelReconciliationAdmissionSchema,
  ModelLoadPlanSchema,
  ModelServingConfigurationIdSchema,
  SlotIdSchema,
  type LocalModelsState,
} from "../schemas/model-state"
import {
  catalogReconciliationIsVisible,
  findLocalModelByConfigurationId,
} from "../schemas/local-model-projection"
import { modelLoadIsVisible, selectedModelStopIsVisible } from "../schemas/model-slot-visibility"
import { ModelSlotSynchronizationFailed, synchronizeModelSlots } from "./configuration"

/**
 * Authoritative local-inference hardware topology. Fresh until the ACN
 * publishes a change for it on `StreamChanges`; retained for the connection lifetime.
 */
const GetLocalInferenceHardware = Query.make("GetLocalInferenceHardware", {
  payload: Schema.Struct({}),
  success: MirroredSnapshotSchema(LocalInferenceHardwareSchema),
  error: Schema.Never,
  staleTime: Infinity,
  gcTime: Infinity,
})

/**
 * Authoritative local-model inventory. Fresh until the ACN publishes a change
 * for it on `StreamChanges`; retained for the connection lifetime.
 */
const GetLocalModels = Query.make("GetLocalModels", {
  payload: Schema.Struct({}),
  success: MirroredSnapshotSchema(LocalModelsStateSchema),
  error: Schema.Never,
  staleTime: Infinity,
  gcTime: Infinity,
})

/** A local-model command was acknowledged but its postcondition did not become visible. */
export class LocalModelSynchronizationFailed extends Data.TaggedError(
  "LocalModelSynchronizationFailed",
)<{ readonly operation: "install" | "cancel" | "delete"; readonly message: string }> {}

/** Reread the inventory after a command acknowledged, returning the fresh state. */
export const synchronizeLocalModels = QueryClient.invalidate(GetLocalModels.match()).pipe(
  Effect.zipRight(QueryClient.fetch(GetLocalModels, {})),
  Effect.map(({ state }) => state),
)

const localModelsSynchronizationSchedule = Schedule.spaced("50 millis").pipe(
  Schedule.intersect(Schedule.recurs(100)),
)

/** Reread until `predicate` holds, bounded by the synchronization schedule. */
const synchronizeLocalModelsUntil = (
  predicate: (state: LocalModelsState) => boolean,
  error: () => LocalModelSynchronizationFailed,
) => synchronizeLocalModels.pipe(
  Effect.filterOrFail(predicate, error),
  Effect.retry(localModelsSynchronizationSchedule),
  Effect.asVoid,
)

const ReconcileCatalogModel = Mutation.make("ReconcileCatalogModel", {
  policy: { recovery: "AtMostOnce" },
  payload: CatalogIdentitySchema,
  success: CatalogModelReconciliationAdmissionSchema,
  error: LocalInferenceError,
  scope: ({ modelId, variantId }) => Mutation.MutationScope(`catalog-model:${modelId}:${variantId}`),
  synchronize: (admission, identity) => synchronizeLocalModelsUntil(
    (state) => catalogReconciliationIsVisible(state, identity, admission),
    () => new LocalModelSynchronizationFailed({
      operation: "install",
      message: "The admitted local-model installation was absent from LocalModels.",
    }),
  ),
})

const CancelModelDownload = Mutation.make("CancelModelDownload", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ downloadId }) => Mutation.MutationScope(`model-download:${downloadId}`),
  synchronize: (_, { downloadId }) => synchronizeLocalModelsUntil(
    (state) => state.models.every((model) => {
      const acquisition = model.acquisitionState
      return acquisition._tag === "NotInstalled"
        || acquisition._tag === "Installed"
        || acquisition.downloadId !== downloadId
        || acquisition._tag === "Cancelled"
        || acquisition._tag === "Failed"
    }),
    () => new LocalModelSynchronizationFailed({
      operation: "cancel",
      message: "The cancelled model download remained active in LocalModels.",
    }),
  ),
})

const DismissModelDownloadFailure = Mutation.make("DismissModelDownloadFailure", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ downloadId: ModelDownloadIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ downloadId }) => Mutation.MutationScope(`model-download:${downloadId}`),
  synchronize: () => synchronizeLocalModels.pipe(Effect.asVoid),
})

const DeleteLocalModel = Mutation.make("DeleteLocalModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ configurationId: ModelServingConfigurationIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ configurationId }) => Mutation.MutationScope(`local-model:${configurationId}`),
  synchronize: (_, { configurationId }) => synchronizeLocalModelsUntil(
    (state) => Option.match(
      findLocalModelByConfigurationId(state.models, configurationId),
      {
        onNone: () => true,
        onSome: (model) => model.acquisitionState._tag !== "Installed",
      },
    ),
    () => new LocalModelSynchronizationFailed({
      operation: "delete",
      message: "The deleted local model remained installed in LocalModels.",
    }),
  ),
})

const LoadModel = Mutation.make("LoadModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-load:${slotId}`),
  synchronize: (_, { slotId }) => synchronizeModelSlots.pipe(
    Effect.filterOrFail(
      ({ state }) => modelLoadIsVisible(state, slotId),
      () => new ModelSlotSynchronizationFailed({
        operation: "load",
        message: "The model load request was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

/** Advisory load plan for a slot; consumers reread it when hardware or slot state changes. */
const PreviewModelLoad = Query.make("PreviewModelLoad", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: ModelLoadPlanSchema,
  error: LocalInferenceError,
})

const StopModel = Mutation.make("StopModel", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
  scope: ({ slotId }) => Mutation.MutationScope(`model-slot-stop:${slotId}`),
  synchronize: (_, { slotId }) => synchronizeModelSlots.pipe(
    Effect.filterOrFail(
      ({ state }) => selectedModelStopIsVisible(state, slotId),
      () => new ModelSlotSynchronizationFailed({
        operation: "stop",
        message: "The selected model remained active in ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

export const LocalInference = Group.make({
  GetLocalInferenceHardware,
  GetLocalModels,
  ReconcileCatalogModel,
  CancelModelDownload,
  DismissModelDownloadFailure,
  DeleteLocalModel,
  LoadModel,
  PreviewModelLoad,
  StopModel,
})
