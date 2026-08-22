import { Rpc } from "@effect/rpc"
import { Data, Effect, Schema } from "effect"
import { Mutation, QueryClient } from "@magnitudedev/effect-query"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
import { Acn } from "../boundary"
import {
  ModelPreferenceMutationFailed,
  ModelSlotUpdateError,
  SessionError,
} from "../errors"
import {
  CloudUsageResponse,
  UsagePeriod,
} from "../schemas/cloud-usage"
import { ProviderAuthSchema } from "../schemas/provider-auth"
import { MirroredSnapshotSchema } from "../schemas/mirrored-state"
import {
  SlotSelectionSchema,
  SlotIdSchema,
  ProviderModelIdentitySchema,
  ProviderModelCatalogStateSchema,
  ModelSlotsStateSchema,
  type SlotId,
} from "../schemas/model-state"
import { slotAssignmentIsVisible } from "../schemas/model-slot-visibility"
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

/**
 * Authoritative slot state. Fresh until the ACN publishes a change for it on
 * `StreamChanges`; retained for the connection lifetime.
 */
export const GetModelSlots = Acn.query("GetModelSlots", {
  payload: Schema.Struct({}),
  success: MirroredSnapshotSchema(ModelSlotsStateSchema),
  error: Schema.Never,
  staleTime: Infinity,
  gcTime: Infinity,
})

/** A slot command was acknowledged but its postcondition did not become visible. */
export class ModelSlotSynchronizationFailed extends Data.TaggedError(
  "ModelSlotSynchronizationFailed",
)<{
  readonly operation: "assign" | "load" | "stop"
  readonly message: string
}> {}

/** Reread slot state after a command acknowledged, returning the fresh snapshot. */
export const synchronizeModelSlots = QueryClient.invalidate(GetModelSlots.match()).pipe(
  Effect.zipRight(QueryClient.fetch(GetModelSlots, {})),
)

export const slotSelectionScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot-selection:${slotId}`)

export const AssignSlot = Acn.mutation("AssignSlot", {
  payload: Schema.Struct({
    slotId: SlotIdSchema,
    selection: SlotSelectionSchema,
  }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotSelectionScope(slotId),
  synchronize: (_, { slotId, selection }) => synchronizeModelSlots.pipe(
    Effect.filterOrFail(
      ({ state }) => slotAssignmentIsVisible(state, slotId, selection),
      () => new ModelSlotSynchronizationFailed({
        operation: "assign",
        message: "The assigned model selection was absent from ModelSlots.",
      }),
    ),
    Effect.asVoid,
  ),
})

export const ClearSlot = Acn.mutation("ClearSlot", {
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotSelectionScope(slotId),
  synchronize: () => synchronizeModelSlots.pipe(Effect.asVoid),
})

export const SetModelFavorite = Acn.mutation("SetModelFavorite", {
  payload: Schema.Struct({
    model: ProviderModelIdentitySchema,
    favorite: Schema.Boolean,
  }),
  success: Schema.Struct({}),
  error: ModelPreferenceMutationFailed,
  scope: ({ model }) => Mutation.MutationScope(`model-favorite:${model.providerId}:${model.providerModelId}`),
  synchronize: () => synchronizeModelSlots.pipe(Effect.asVoid),
})
