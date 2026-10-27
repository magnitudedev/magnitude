import { Effect, Schedule } from "effect";
import { Models as Rpcs } from "@magnitudedev/sdk";
import { Group, Mutation, QueryClient } from "@magnitudedev/effect-query";
import { type CandidateInspectionUpdate, type SlotId } from "@magnitudedev/sdk";
import { mutation, query, streamQuery } from "./bind";

const GetServingUsage = query(Rpcs.getServingUsage, client => client.models.getServingUsage, {
  staleTime: 0, refresh: Schedule.spaced("5 seconds"),
});

const ListServableModels = query(
  Rpcs.listServableModels,
  (client) => client.models.listServableModels,
  { staleTime: "10 seconds" }
);

const GetCatalog = query(
  Rpcs.getCatalog,
  (client) => client.models.getCatalog,
  { staleTime: Infinity, gcTime: Infinity }
);

const GetSlots = query(Rpcs.getSlots, (client) => client.models.getSlots, {
  staleTime: Infinity,
  gcTime: Infinity,
});

const GetLocalEnvironment = query(
  Rpcs.getLocalEnvironment,
  (client) => client.models.getLocalEnvironment,
  { staleTime: Infinity, gcTime: Infinity }
);

const RefreshCatalog = mutation(
  Rpcs.refreshCatalog,
  (client) => client.models.refreshCatalog,
  { synchronize: () => QueryClient.refetch(GetCatalog.match()) }
);

const slotScope = (slotId: SlotId) =>
  Mutation.MutationScope(`model-slot:${slotId}`);

const synchronizeCatalog = QueryClient.refetch(GetCatalog.match());

const synchronizeSlots = QueryClient.refetch(GetSlots.match());

const AssignSlot = mutation(
  Rpcs.assignSlot,
  (client) => client.models.assignSlot,
  {
    scope: ({ slotId }) => slotScope(slotId),
    synchronize: () => synchronizeSlots,
  }
);

const ClearSlot = mutation(
  Rpcs.clearSlot,
  (client) => client.models.clearSlot,
  {
    scope: ({ slotId }) => slotScope(slotId),
    synchronize: () => synchronizeSlots,
  }
);

const SetFavorite = mutation(
  Rpcs.setFavorite,
  (client) => client.models.setFavorite,
  {
    scope: ({ model }) =>
      Mutation.MutationScope(
        `model-favorite:${model.providerId}:${model.providerModelId}`
      ),
    synchronize: () => synchronizeSlots,
  }
);

const SyncLocalModel = mutation(
  Rpcs.syncLocalModel,
  (client) => client.models.syncLocalModel,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const DownloadCatalogComponent = mutation(
  Rpcs.downloadCatalogComponent,
  (client) => client.models.downloadCatalogComponent,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const CancelLocalModelSync = mutation(
  Rpcs.cancelLocalModelSync,
  (client) => client.models.cancelLocalModelSync,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const AcknowledgeLocalModelSyncFailure = mutation(
  Rpcs.acknowledgeLocalModelSyncFailure,
  (client) => client.models.acknowledgeLocalModelSyncFailure,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const RemoveLocalModel = mutation(
  Rpcs.removeLocalModel,
  (client) => client.models.removeLocalModel,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const LoadLocalModel = mutation(Rpcs.load, (client) => client.models.load, {
  scope: ({ modelId }) =>
    Mutation.MutationScope(`local-model-residency:${modelId}`),
  synchronize: () => synchronizeCatalog,
});

const StopActiveLocalModel = mutation(
  Rpcs.stop,
  (client) => client.models.stop,
  {
    scope: () => Mutation.MutationScope("active-local-model-residency"),
    synchronize: () => synchronizeCatalog,
  }
);

const LoadSlot = mutation(Rpcs.loadSlot, (client) => client.models.loadSlot, {
  scope: ({ slotId }) =>
    Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => synchronizeSlots,
});

const StopSlot = mutation(Rpcs.stopSlot, (client) => client.models.stopSlot, {
  scope: ({ slotId }) =>
    Mutation.MutationScope(`model-slot-instance:${slotId}`),
  synchronize: () => synchronizeSlots,
});

const PreviewSlotLoad = query(
  Rpcs.previewSlotLoad,
  (client) => client.models.previewSlotLoad
);

const SearchHuggingFaceModels = query(
  Rpcs.searchHuggingFaceModels,
  (client) => client.models.searchHuggingFaceModels,
  { staleTime: "1 minute" }
);

/**
 * The same inspection as it goes: the latest progress, then the inspection, which stays cached so
 * reopening the repository shows it at once.
 */
const StreamModelCandidateInspection = streamQuery(
  Rpcs.streamModelCandidateInspection,
  (client) => client.models.streamModelCandidateInspection,
  { reduce: (_, update): CandidateInspectionUpdate => update, gcTime: "5 minutes" }
);

const ListDrafterCandidates = query(
  Rpcs.listDrafterCandidates,
  (client) => client.models.listDrafterCandidates,
  { staleTime: "1 minute" }
);

const ListModelLocations = query(
  Rpcs.listModelLocations,
  (client) => client.models.listModelLocations,
  { staleTime: Infinity }
);

const synchronizeLocations = QueryClient.refetch(ListModelLocations.match());

const AddHuggingFaceModel = mutation(
  Rpcs.addHuggingFaceModel,
  (client) => client.models.addHuggingFaceModel,
  {
    scope: ({ repository }) => Mutation.MutationScope(`local-model-source:hf:${repository}`),
    synchronize: () => synchronizeCatalog,
  }
);

const SetModelDrafter = mutation(
  Rpcs.setModelDrafter,
  (client) => client.models.setModelDrafter,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const HideModel = mutation(
  Rpcs.hideModel,
  (client) => client.models.hideModel,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const UnhideModel = mutation(
  Rpcs.unhideModel,
  (client) => client.models.unhideModel,
  {
    scope: ({ modelId }) => Mutation.MutationScope(`local-model:${modelId}`),
    synchronize: () => synchronizeCatalog,
  }
);

const modelLocationsScope = Mutation.MutationScope("model-locations");

const synchronizeLocationsAndCatalog = Effect.all([synchronizeLocations, synchronizeCatalog], { discard: true });

const AddModelFolder = mutation(
  Rpcs.addModelFolder,
  (client) => client.models.addModelFolder,
  { scope: () => modelLocationsScope, synchronize: () => synchronizeLocationsAndCatalog }
);

const RemoveModelFolder = mutation(
  Rpcs.removeModelFolder,
  (client) => client.models.removeModelFolder,
  { scope: () => modelLocationsScope, synchronize: () => synchronizeLocationsAndCatalog }
);

const SetModelLocationEnabled = mutation(
  Rpcs.setModelLocationEnabled,
  (client) => client.models.setModelLocationEnabled,
  { scope: () => modelLocationsScope, synchronize: () => synchronizeLocationsAndCatalog }
);

export const Models = Group.make({
  GetServingUsage,
  ListServableModels,
  GetCatalog,
  GetSlots,
  GetLocalEnvironment,
  PreviewSlotLoad,
  SearchHuggingFaceModels,
  StreamModelCandidateInspection,
  ListDrafterCandidates,
  ListModelLocations,
  RefreshCatalog,
  AssignSlot,
  ClearSlot,
  SetFavorite,
  SyncLocalModel,
  DownloadCatalogComponent,
  CancelLocalModelSync,
  AcknowledgeLocalModelSyncFailure,
  RemoveLocalModel,
  LoadLocalModel,
  StopActiveLocalModel,
  LoadSlot,
  StopSlot,
  AddHuggingFaceModel,
  SetModelDrafter,
  HideModel,
  UnhideModel,
  AddModelFolder,
  RemoveModelFolder,
  SetModelLocationEnabled,
});
