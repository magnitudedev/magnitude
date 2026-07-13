import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import {
  BalanceResponse,
  UsagePeriod,
  SlotId,
  SlotProfiles,
  SlotModelConfigSchema,
  ModelListSchema,
  ProviderAuthSchema,
} from "../schemas/account"

export const UpdateProviderAuth = Rpc.make("UpdateProviderAuth", {
  payload: Schema.Struct({
    providerId: Schema.String,
    auth: ProviderAuthSchema,
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const GetProviderAuth = Rpc.make("GetProviderAuth", {
  payload: Schema.Struct({
    providerId: Schema.String,
  }),
  success: Schema.Struct({
    auth: Schema.optionalWith(ProviderAuthSchema, { as: "Option", exact: true }),
  }),
  error: SessionError
})

export const ListProviderAuth = Rpc.make("ListProviderAuth", {
  payload: Schema.Struct({}),
  success: Schema.Struct({
    auths: Schema.Record({ key: Schema.String, value: ProviderAuthSchema }),
  }),
  error: SessionError
})

export const GetBalance = Rpc.make("GetBalance", {
  payload: Schema.Struct({
    period: Schema.optional(UsagePeriod),
    days: Schema.optional(Schema.Number),
    tz: Schema.optional(Schema.String),
  }),
  success: BalanceResponse,
  error: SessionError
})

// ── Slot-based model configuration ──

export const ListPublicSlotProfiles = Rpc.make("ListPublicSlotProfiles", {
  payload: Schema.Struct({}),
  success: Schema.NullOr(SlotProfiles),
  error: SessionError
})

export const UpdateModelConfig = Rpc.make("UpdateModelConfig", {
  payload: Schema.Struct({
    slots: Schema.partial(Schema.Record({ key: SlotId, value: SlotModelConfigSchema })),
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const GetCachedModelList = Rpc.make("GetCachedModelList", {
  payload: Schema.Struct({
    providerId: Schema.optional(Schema.String),
  }),
  success: ModelListSchema,
  error: SessionError
})

export const RefreshCachedModelList = Rpc.make("RefreshCachedModelList", {
  payload: Schema.Struct({
    providerId: Schema.optional(Schema.String),
  }),
  success: ModelListSchema,
  error: SessionError
})
