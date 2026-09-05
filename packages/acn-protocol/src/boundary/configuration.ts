import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
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

export const Configuration = {
  getProviderAuth: GetProviderAuth,
  listProviderAuth: ListProviderAuth,
  updateProviderAuth: UpdateProviderAuth,
  getCloudUsage: GetCloudUsage,
}
