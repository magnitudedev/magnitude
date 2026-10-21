import { ServingUsageRequest, ServingUsageSnapshot } from "../schemas/serving-usage"
import { Rpc } from "@effect/rpc"
import { replaySafe, atMostOnce } from "../transport/recovery"
import { Option, Schema } from "effect"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import {
  CustomModelMutationError,
  LocalInferenceError,
  LocalModelMutationFailed,
  ModelCandidateError,
  ModelLocationError,
  ModelPreferenceMutationFailed,
  ModelSlotUpdateError,
  ModelVersionNotRunnable,
} from "../errors"
import {
  AutomaticModelLocationKindSchema,
  AddedModelSchema,
  CatalogComponentSchema,
  CandidateInspectionSchema,
  CandidateInspectionUpdateSchema,
  CustomFormModelIdSchema,
  DrafterCandidateSchema,
  DrafterChoiceSchema,
  DrafterTargetSchema,
  HuggingFaceRepositoryIdSchema,
  HuggingFaceSearchResponseSchema,
  LocalInferenceHardwareSchema,
  LocalModelSyncOutcomeSchema,
  ModelIdSchema,
  ModelLocationSchema,
  ModelCatalogStateSchema,
  ModelLoadPlanSchema,
  ModelSlotsStateSchema,
  ProviderModelIdentitySchema,
  SlotIdSchema,
  SlotSelectionSchema,
} from "../schemas/model-state"

const GetServingUsage = Rpc.make("GetServingUsage", { payload: ServingUsageRequest, success: ServingUsageSnapshot }).pipe(replaySafe)

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

/** A catalog model not yet on this computer that needs more memory than it can provide is refused (ModelVersionNotRunnable). */
const SyncLocalModel = Rpc.make("SyncLocalModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({ outcome: LocalModelSyncOutcomeSchema }),
  error: Schema.Union(LocalInferenceError, ModelVersionNotRunnable),
}).pipe(atMostOnce)

/** Fetches only one component a catalog model lacks (see CatalogMissingComponent); false when it is already here. */
const DownloadCatalogComponent = Rpc.make("DownloadCatalogComponent", {
  payload: Schema.Struct({ modelId: ModelIdSchema, component: CatalogComponentSchema }),
  success: Schema.Struct({ started: Schema.Boolean }),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const CancelLocalModelSync = Rpc.make("CancelLocalModelSync", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const AcknowledgeLocalModelSyncFailure = Rpc.make(
  "AcknowledgeLocalModelSyncFailure",
  {
    payload: Schema.Struct({ modelId: ModelIdSchema }),
    success: Schema.Struct({}),
    error: LocalInferenceError,
  }
).pipe(atMostOnce)

/** `keptComponents`: the parts Magnitude stores for a found catalog model that the removal kept for another model. */
const RemoveLocalModel = Rpc.make("RemoveLocalModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({ keptComponents: Schema.Array(CatalogComponentSchema) }),
  error: LocalInferenceError,
}).pipe(atMostOnce)

const LoadLocalModel = Rpc.make("LoadLocalModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({}),
  error: Schema.Union(LocalInferenceError, ModelVersionNotRunnable),
}).pipe(atMostOnce)

const StopActiveLocalModel = Rpc.make("StopActiveLocalModel", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: LocalInferenceError,
}).pipe(atMostOnce)

/** `stopped` is false when the model wasn't loading, running or stopping, so there was nothing to stop. */
const StopLocalModel = Rpc.make("StopLocalModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({ stopped: Schema.Boolean }),
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

const optional = <A, I, R>(schema: Schema.Schema<A, I, R>) =>
  Schema.optionalWith(schema, { as: "Option", exact: true })

const NonEmptyString = Schema.String.pipe(Schema.minLength(1))

/** GGUF repositories on Hugging Face; those of architectures Magnitude doesn't run carry their reasons. */
const SearchHuggingFaceModels = Rpc.make("SearchHuggingFaceModels", {
  payload: Schema.Struct({
    query: Schema.String,
    limit: optional(Schema.Int.pipe(Schema.between(1, 100))),
  }),
  success: HuggingFaceSearchResponseSchema,
  error: ModelCandidateError,
}).pipe(replaySafe)

/** A Hugging Face URL, `owner/repo`, `owner/repo:QUANT`, or local GGUF path, inspected without changes. */
const InspectModelCandidate = Rpc.make("InspectModelCandidate", {
  payload: Schema.Struct({ reference: NonEmptyString }),
  success: CandidateInspectionSchema,
  error: ModelCandidateError,
}).pipe(replaySafe)

/**
 * The same inspection as it goes: its progress in order, then the inspection itself. Like every ACN
 * subscription, it stays open after the inspection until the subscriber leaves.
 */
const StreamModelCandidateInspection = Rpc.make("StreamModelCandidateInspection", {
  payload: Schema.Struct({ reference: NonEmptyString }),
  success: CandidateInspectionUpdateSchema,
  error: ModelCandidateError,
  stream: true,
})

/** Drafters for a target in preselection order; `other` adds any reference with the engine's verdict. */
/** A target's drafter choices; with `other`, also the listed candidate that different drafter is. */
const ListDrafterCandidates = Rpc.make("ListDrafterCandidates", {
  payload: Schema.Struct({
    target: DrafterTargetSchema,
    other: optional(NonEmptyString),
  }),
  success: Schema.Struct({
    candidates: Schema.Array(DrafterCandidateSchema),
    entered: optional(DrafterCandidateSchema),
  }),
  error: ModelCandidateError,
}).pipe(replaySafe)

const ListModelLocations = Rpc.make("ListModelLocations", {
  payload: Schema.Struct({}),
  success: Schema.Struct({ locations: Schema.Array(ModelLocationSchema) }),
  error: LocalModelMutationFailed,
}).pipe(replaySafe)

/**
 * Pins the repository's commit, picks the recommended version when none is named, and starts the download. A
 * version the engine can't run is refused, and one that needs more memory than this computer can provide is
 * refused unless `force` is set (ModelVersionNotRunnable). A version that is exactly a catalog model is never
 * declared as a custom model: the catalog model is installed instead, with its own drafter.
 */
const AddHuggingFaceModel = Rpc.make("AddHuggingFaceModel", {
  payload: Schema.Struct({
    repository: HuggingFaceRepositoryIdSchema,
    revision: optional(NonEmptyString),
    version: optional(NonEmptyString),
    drafter: DrafterChoiceSchema,
    force: Schema.Boolean,
  }),
  success: AddedModelSchema,
  error: CustomModelMutationError,
}).pipe(atMostOnce)

/** Adds, changes, or turns off a custom model's drafter; downloads only the drafter when needed. */
const SetModelDrafter = Rpc.make("SetModelDrafter", {
  payload: Schema.Struct({ modelId: CustomFormModelIdSchema, drafter: DrafterChoiceSchema }),
  success: Schema.Struct({ drafter: optional(DrafterCandidateSchema), drafterDownloadStarted: Schema.Boolean }),
  error: CustomModelMutationError,
}).pipe(atMostOnce)

const HideModel = Rpc.make("HideModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalModelMutationFailed,
}).pipe(atMostOnce)

const UnhideModel = Rpc.make("UnhideModel", {
  payload: Schema.Struct({ modelId: ModelIdSchema }),
  success: Schema.Struct({}),
  error: LocalModelMutationFailed,
}).pipe(atMostOnce)

const AddModelFolder = Rpc.make("AddModelFolder", {
  payload: Schema.Struct({ path: NonEmptyString }),
  success: Schema.Struct({}),
  error: ModelLocationError,
}).pipe(atMostOnce)

const RemoveModelFolder = Rpc.make("RemoveModelFolder", {
  payload: Schema.Struct({ path: NonEmptyString }),
  success: Schema.Struct({}),
  error: ModelLocationError,
}).pipe(atMostOnce)

const SetModelLocationEnabled = Rpc.make("SetModelLocationEnabled", {
  payload: Schema.Struct({ kind: AutomaticModelLocationKindSchema, enabled: Schema.Boolean }),
  success: Schema.Struct({}),
  error: LocalModelMutationFailed,
}).pipe(atMostOnce)

export const Models = {
  getServingUsage: GetServingUsage,
  getCatalog: GetCatalog,
  getSlots: GetSlots,
  getLocalEnvironment: GetLocalEnvironment,
  refreshCatalog: RefreshCatalog,
  assignSlot: AssignSlot,
  clearSlot: ClearSlot,
  setFavorite: SetFavorite,
  syncLocalModel: SyncLocalModel,
  downloadCatalogComponent: DownloadCatalogComponent,
  cancelLocalModelSync: CancelLocalModelSync,
  acknowledgeLocalModelSyncFailure: AcknowledgeLocalModelSyncFailure,
  removeLocalModel: RemoveLocalModel,
  load: LoadLocalModel,
  stop: StopActiveLocalModel,
  stopModel: StopLocalModel,
  loadSlot: LoadSlot,
  stopSlot: StopSlot,
  previewSlotLoad: PreviewSlotLoad,
  searchHuggingFaceModels: SearchHuggingFaceModels,
  inspectModelCandidate: InspectModelCandidate,
  streamModelCandidateInspection: StreamModelCandidateInspection,
  listDrafterCandidates: ListDrafterCandidates,
  listModelLocations: ListModelLocations,
  addHuggingFaceModel: AddHuggingFaceModel,
  setModelDrafter: SetModelDrafter,
  hideModel: HideModel,
  unhideModel: UnhideModel,
  addModelFolder: AddModelFolder,
  removeModelFolder: RemoveModelFolder,
  setModelLocationEnabled: SetModelLocationEnabled,
}
