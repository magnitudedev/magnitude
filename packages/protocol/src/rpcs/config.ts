import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProviderIdSchema } from "@magnitudedev/ai"
import { SessionError } from "../errors"
import {
  CloudUsageResponse,
  UsagePeriod,
  SlotId,
  SlotProfiles,
  SlotModelConfigSchema,
  ModelCatalogSchema,
  ModelSlotsSchema,
  ProviderAuthSchema,
} from "../schemas/account"
import { MirroredResourceInvalidationSchema } from "../schemas/mirrored-resource"
import { StreamHeartbeat } from "../schemas/events"

export const UpdateProviderAuth = Rpc.make("UpdateProviderAuth", {
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
    auth: ProviderAuthSchema,
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const GetProviderAuth = Rpc.make("GetProviderAuth", {
  payload: Schema.Struct({
    providerId: ProviderIdSchema,
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

export const GetCloudUsage = Rpc.make("GetCloudUsage", {
  payload: Schema.Struct({
    period: Schema.optional(UsagePeriod),
    days: Schema.optional(Schema.Number),
    tz: Schema.optional(Schema.String),
  }),
  success: CloudUsageResponse,
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
  success: Schema.Union(MirroredResourceInvalidationSchema, StreamHeartbeat),
  error: SessionError,
  stream: true,
})

export const RefreshModelCatalog = Rpc.make("RefreshModelCatalog", {
  payload: Schema.Struct({
    providerId: Schema.optionalWith(ProviderIdSchema, { as: "Option", exact: true }),
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
  success: Schema.Union(MirroredResourceInvalidationSchema, StreamHeartbeat),
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
