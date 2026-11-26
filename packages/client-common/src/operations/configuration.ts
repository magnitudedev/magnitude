import { Configuration as Rpcs } from "@magnitudedev/sdk";
import { Group, QueryClient } from "@magnitudedev/effect-query";
import { mutation, query } from "./bind";

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
const GetOnboarding = query(Rpcs.getOnboarding, (client) => client.configuration.getOnboarding);
const CompleteOnboarding = mutation(Rpcs.completeOnboarding, (client) => client.configuration.completeOnboarding, {
  synchronize: () => QueryClient.invalidate(GetOnboarding.match({})),
});

export const Configuration = Group.make({
  GetModelStorage,
  SetModelStorage,
  BrowseDirectories,
  GetNetworkAccess,
  SetNetworkAccess,
  RegenerateNetworkApiKey,
  GetServerMachine,
  GetOnboarding,
  CompleteOnboarding,
});
