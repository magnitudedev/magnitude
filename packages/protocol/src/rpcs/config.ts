import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { SessionError } from "../errors"
import {
  BalanceResponse,
  UsagePeriod,
  SlotId,
  SlotProfiles,
  SlotModelConfigSchema,
  ModelCatalogSchema,
  ModelSlotsSchema,
  ModelResourceInvalidationSchema,
  ProviderAuthSchema,
} from "../schemas/account"
import { StreamHeartbeat } from "../schemas/events"

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

export const GetModelCatalog = Rpc.make("GetModelCatalog", {
  payload: Schema.Struct({}),
  success: ModelCatalogSchema,
  error: SessionError,
})

export const WatchModelCatalog = Rpc.make("WatchModelCatalog", {
  payload: Schema.Struct({}),
  success: Schema.Union(ModelResourceInvalidationSchema, StreamHeartbeat),
  error: SessionError,
  stream: true,
})

export const RefreshModelCatalog = Rpc.make("RefreshModelCatalog", {
  payload: Schema.Struct({
    providerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: SessionError,
})

export const GetModelSlots = Rpc.make("GetModelSlots", {
  payload: Schema.Struct({}),
  success: ModelSlotsSchema,
  error: SessionError,
})

export const WatchModelSlots = Rpc.make("WatchModelSlots", {
  payload: Schema.Struct({}),
  success: Schema.Union(ModelResourceInvalidationSchema, StreamHeartbeat),
  error: SessionError,
  stream: true,
})

export const UpdateModelSlots = Rpc.make("UpdateModelSlots", {
  payload: Schema.Struct({
    slots: Schema.partial(Schema.Record({ key: SlotId, value: SlotModelConfigSchema })),
  }),
  success: Schema.Struct({}),
  error: SessionError,
})
