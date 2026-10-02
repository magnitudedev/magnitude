import { Rpc } from "@effect/rpc"
import { atMostOnce, replaySafe } from "../transport/recovery"
import { Schema } from "effect"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import {
  SessionError,
} from "../errors"
import {
  CloudUsageResponse,
  UsagePeriod,
} from "../schemas/cloud-usage"
import { ProviderAuthSchema } from "../schemas/provider-auth"
import {
  DirectoryListing,
  ModelStorageChange,
  ModelStorageSettings,
  NetworkAccessChange,
  NetworkAccessSettings,
  ServerMachine,
  ServerSettingsFailed,
} from "../schemas/server-settings"

const GetProviderAuth = Rpc.make("GetProviderAuth", {
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
  }),
  success: Schema.Struct({
    auth: Schema.optionalWith(ProviderAuthSchema, { as: "Option", exact: true }),
  }),
  error: SessionError,
}).pipe(replaySafe)

const ListProviderAuth = Rpc.make("ListProviderAuth", {
  payload: Schema.Struct({}),
  success: Schema.Struct({
    auths: Schema.Record({ key: ProviderIdSchema, value: ProviderAuthSchema }),
  }),
  error: SessionError,
}).pipe(replaySafe)

const UpdateProviderAuth = Rpc.make("UpdateProviderAuth", {
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
    auth: ProviderAuthSchema,
  }),
  success: Schema.Struct({}),
  error: SessionError,
}).pipe(replaySafe)

const GetCloudUsage = Rpc.make("GetCloudUsage", {
  payload: Schema.Struct({
    period: Schema.optional(UsagePeriod),
    days: Schema.optional(Schema.Number),
    tz: Schema.optional(Schema.String),
  }),
  success: CloudUsageResponse,
  error: SessionError,
}).pipe(replaySafe)

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

/** Child directories on the machine running Magnitude, for choosing the model folder there. */
const BrowseDirectories = Rpc.make("BrowseDirectories", {
  payload: Schema.Struct({ path: Schema.optionalWith(Schema.String, { as: "Option", exact: true }) }),
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

export const Configuration = {
  getProviderAuth: GetProviderAuth,
  listProviderAuth: ListProviderAuth,
  updateProviderAuth: UpdateProviderAuth,
  getCloudUsage: GetCloudUsage,
  getModelStorage: GetModelStorage,
  setModelStorage: SetModelStorage,
  browseDirectories: BrowseDirectories,
  getNetworkAccess: GetNetworkAccess,
  setNetworkAccess: SetNetworkAccess,
  regenerateNetworkApiKey: RegenerateNetworkApiKey,
  getServerMachine: GetServerMachine,
}
