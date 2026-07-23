import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import { ModelSlotUpdateError, SessionError } from "../errors"
import {
  CloudUsageResponse,
  UsagePeriod,
} from "../schemas/cloud-usage"
import { ProviderAuthSchema } from "../schemas/provider-auth"
import {
  SlotSelectionSchema,
  SlotIdSchema,
  ProviderModelCatalogStateSchema,
  ModelSlotsStateSchema,
} from "../schemas/model-state"
import { defineMirroredState } from "./mirrored-state"

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
    auths: Schema.Record({ key: ProviderIdSchema, value: ProviderAuthSchema }),
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

export const ProviderModelCatalogMirror = defineMirroredState("GetProviderModelCatalog", {
  stateSchema: ProviderModelCatalogStateSchema,
  errorSchema: Schema.Never,
})

export const RefreshModelCatalog = Rpc.make("RefreshModelCatalog", {
  payload: Schema.Struct({
    providerId: Schema.optionalWith(ProviderIdSchema, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: Schema.Never,
})

export const ModelSlotsMirror = defineMirroredState("GetModelSlots", {
  stateSchema: ModelSlotsStateSchema,
  errorSchema: Schema.Never,
})

export const AssignSlot = Rpc.make("AssignSlot", {
  payload: Schema.Struct({
    slotId: SlotIdSchema,
    selection: SlotSelectionSchema,
  }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
})

export const ClearSlot = Rpc.make("ClearSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
})
