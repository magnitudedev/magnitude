import { Models as Rpcs } from "@magnitudedev/sdk";
import { Group, Mutation, QueryClient } from "@magnitudedev/effect-query";
import { type SlotId } from "@magnitudedev/sdk";
import { turnAdmissionScope } from "./configuration";
import { mutation, query } from "./bind";

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

const slotMutationScope = (slotId: SlotId) =>
  slotId === "primary" ? turnAdmissionScope : slotScope(slotId);

const synchronizeCatalog = QueryClient.refetch(GetCatalog.match());

const synchronizeSlots = QueryClient.refetch(GetSlots.match());

const AssignSlot = mutation(
  Rpcs.assignSlot,
  (client) => client.models.assignSlot,
  {
    scope: ({ slotId }) => slotMutationScope(slotId),
    synchronize: () => synchronizeSlots,
  }
);

const ClearSlot = mutation(
  Rpcs.clearSlot,
  (client) => client.models.clearSlot,
  {
    scope: ({ slotId }) => slotMutationScope(slotId),
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
  LoadLocalModel,
  StopActiveLocalModel,
  LoadSlot,
  StopSlot,
});
