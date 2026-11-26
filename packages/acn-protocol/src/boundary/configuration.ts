import { Rpc } from "@effect/rpc"
import { atMostOnce, replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import {
  DirectoryListing,
  ModelStorageChange,
  ModelStorageSettings,
  NetworkAccessChange,
  NetworkAccessSettings,
  OnboardingSettings,
  ServerMachine,
  ServerSettingsFailed,
} from "../schemas/server-settings"

const GetModelStorage = Rpc.make("GetModelStorage", {
  payload: Schema.Struct({}),
  success: ModelStorageSettings,
  error: ServerSettingsFailed,
}).pipe(replaySafe)

/** Saves the folder the engine uses after the next service start; `None` restores the default. */
const SetModelStorage = Rpc.make("SetModelStorage", {
  payload: ModelStorageChange,
  success: Schema.Struct({}),
  error: ServerSettingsFailed,
}).pipe(atMostOnce)

/** Child directories on the machine running Magnitude, for choosing a model folder. */
const BrowseDirectories = Rpc.make("BrowseDirectories", {
  payload: Schema.Struct({
    path: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: DirectoryListing,
  error: ServerSettingsFailed,
}).pipe(replaySafe)

const GetNetworkAccess = Rpc.make("GetNetworkAccess", {
  payload: Schema.Struct({}),
  success: NetworkAccessSettings,
  error: ServerSettingsFailed,
}).pipe(replaySafe)

const SetNetworkAccess = Rpc.make("SetNetworkAccess", {
  payload: NetworkAccessChange,
  success: Schema.Struct({}),
  error: ServerSettingsFailed,
}).pipe(atMostOnce)

const RegenerateNetworkApiKey = Rpc.make("RegenerateNetworkApiKey", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: ServerSettingsFailed,
}).pipe(atMostOnce)

const GetServerMachine = Rpc.make("GetServerMachine", {
  payload: Schema.Struct({}),
  success: ServerMachine,
}).pipe(replaySafe)

const GetOnboarding = Rpc.make("GetOnboarding", {
  payload: Schema.Struct({}),
  success: OnboardingSettings,
  error: ServerSettingsFailed,
}).pipe(replaySafe)

/** Marks first-run onboarding finished; completing it again changes nothing. */
const CompleteOnboarding = Rpc.make("CompleteOnboarding", {
  payload: Schema.Struct({}),
  success: Schema.Struct({}),
  error: ServerSettingsFailed,
}).pipe(replaySafe)

export const Configuration = {
  getModelStorage: GetModelStorage,
  setModelStorage: SetModelStorage,
  browseDirectories: BrowseDirectories,
  getNetworkAccess: GetNetworkAccess,
  setNetworkAccess: SetNetworkAccess,
  regenerateNetworkApiKey: RegenerateNetworkApiKey,
  getServerMachine: GetServerMachine,
  getOnboarding: GetOnboarding,
  completeOnboarding: CompleteOnboarding,
}
