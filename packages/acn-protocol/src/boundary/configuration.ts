import { Effect, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import {
  SessionError,
} from "../errors"
import {
  CloudUsageResponse,
  UsagePeriod,
} from "../schemas/cloud-usage"
import { ProviderAuthSchema } from "../schemas/provider-auth"

const GetProviderAuth = Query.make("GetProviderAuth", {
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
  }),
  success: Schema.Struct({
    auth: Schema.optionalWith(ProviderAuthSchema, { as: "Option", exact: true }),
  }),
  error: SessionError,
})

const ListProviderAuth = Query.make("ListProviderAuth", {
  payload: Schema.Struct({}),
  success: Schema.Struct({
    auths: Schema.Record({ key: ProviderIdSchema, value: ProviderAuthSchema }),
  }),
  error: SessionError,
})

/** Provider auth is not poked by the ACN; the command rereads what it changed. */
const UpdateProviderAuth = Mutation.make("UpdateProviderAuth", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
    auth: ProviderAuthSchema,
  }),
  success: Schema.Struct({}),
  error: SessionError,
  synchronize: (_, { providerId }) => QueryClient.invalidate(GetProviderAuth.match({ providerId })).pipe(
    Effect.zipRight(QueryClient.invalidate(ListProviderAuth.match())),
  ),
})

const GetCloudUsage = Query.make("GetCloudUsage", {
  payload: Schema.Struct({
    period: Schema.optional(UsagePeriod),
    days: Schema.optional(Schema.Number),
    tz: Schema.optional(Schema.String),
  }),
  success: CloudUsageResponse,
  error: SessionError,
})

/**
 * Primary-model selection and work admission share one serialization boundary:
 * a turn must never observe the selection before its mutation has synchronized.
 */
export const turnAdmissionScope = Mutation.MutationScope("turn-admission")

export const Configuration = Group.make({
  GetProviderAuth,
  ListProviderAuth,
  UpdateProviderAuth,
  GetCloudUsage,
})
