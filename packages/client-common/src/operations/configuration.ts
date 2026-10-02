import { Configuration as Rpcs } from "@magnitudedev/sdk";
import { Effect } from "effect";
import { Group, Mutation, QueryClient } from "@magnitudedev/effect-query";
import { mutation, query } from "./bind";

const GetProviderAuth = query(
  Rpcs.getProviderAuth,
  (client) => client.configuration.getProviderAuth
);

const ListProviderAuth = query(
  Rpcs.listProviderAuth,
  (client) => client.configuration.listProviderAuth
);

const UpdateProviderAuth = mutation(
  Rpcs.updateProviderAuth,
  (client) => client.configuration.updateProviderAuth,
  {
    synchronize: (_, { providerId }) =>
      QueryClient.invalidate(GetProviderAuth.match({ providerId })).pipe(
        Effect.zipRight(QueryClient.invalidate(ListProviderAuth.match()))
      ),
  }
);

const GetCloudUsage = query(
  Rpcs.getCloudUsage,
  (client) => client.configuration.getCloudUsage
);

/**
 * Primary-model selection and work admission share one serialization boundary:
 * a turn must never observe the selection before its mutation has synchronized.
 */
export const turnAdmissionScope = Mutation.MutationScope("turn-admission");

const GetModelStorage = query(Rpcs.getModelStorage, (client) => client.configuration.getModelStorage);
const SetModelStorage = mutation(Rpcs.setModelStorage, (client) => client.configuration.setModelStorage, {
  synchronize: () => QueryClient.invalidate(GetModelStorage.match({})),
});
const BrowseDirectories = query(Rpcs.browseDirectories, (client) => client.configuration.browseDirectories);
const GetNetworkAccess = query(Rpcs.getNetworkAccess, (client) => client.configuration.getNetworkAccess);
const SetNetworkAccess = mutation(Rpcs.setNetworkAccess, (client) => client.configuration.setNetworkAccess, {
  synchronize: () => QueryClient.invalidate(GetNetworkAccess.match({})),
});
const RegenerateNetworkApiKey = mutation(Rpcs.regenerateNetworkApiKey, (client) => client.configuration.regenerateNetworkApiKey, {
  synchronize: () => QueryClient.invalidate(GetNetworkAccess.match({})),
});
const GetServerMachine = query(Rpcs.getServerMachine, (client) => client.configuration.getServerMachine);

export const Configuration = Group.make({
  GetProviderAuth,
  ListProviderAuth,
  UpdateProviderAuth,
  GetCloudUsage,
  GetModelStorage,
  SetModelStorage,
  BrowseDirectories,
  GetNetworkAccess,
  SetNetworkAccess,
  RegenerateNetworkApiKey,
  GetServerMachine,
});
