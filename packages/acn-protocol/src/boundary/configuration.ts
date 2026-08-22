import { Data, Effect, Schema } from "effect"
import { Group, Mutation, Query, QueryClient } from "@magnitudedev/effect-query"
import { ProviderIdSchema } from "@magnitudedev/ai/provider/model"
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

// ── Model catalog and slot-based model configuration ──

/**
 * Authoritative provider model catalog. Fresh until the ACN publishes a
 * change for it on `StreamChanges`; retained for the connection lifetime.
 */
const GetProviderModelCatalog = Query.make("GetProviderModelCatalog", {
  payload: Schema.Struct({}),
  success: MirroredSnapshotSchema(ProviderModelCatalogStateSchema),
  error: Schema.Never,
  staleTime: Infinity,
  gcTime: Infinity,
})

/** Catalog refresh publishes through the catalog's own change poke. */
const RefreshModelCatalog = Mutation.make("RefreshModelCatalog", {
  policy: { recovery: "AtMostOnce" },
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
const GetModelSlots = Query.make("GetModelSlots", {
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

const slotSelectionScope = (slotId: SlotId): Mutation.MutationScope =>
  Mutation.MutationScope(`model-slot-selection:${slotId}`)

const AssignSlot = Mutation.make("AssignSlot", {
  policy: { recovery: "ReplaySafe" },
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

const ClearSlot = Mutation.make("ClearSlot", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ slotId: SlotIdSchema }),
  success: Schema.Struct({}),
  error: ModelSlotUpdateError,
  scope: ({ slotId }) => slotSelectionScope(slotId),
  synchronize: () => synchronizeModelSlots.pipe(Effect.asVoid),
})

const SetModelFavorite = Mutation.make("SetModelFavorite", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({
    model: ProviderModelIdentitySchema,
    favorite: Schema.Boolean,
  }),
  success: Schema.Struct({}),
  error: ModelPreferenceMutationFailed,
  scope: ({ model }) => Mutation.MutationScope(`model-favorite:${model.providerId}:${model.providerModelId}`),
  synchronize: () => synchronizeModelSlots.pipe(Effect.asVoid),
})

export const Configuration = Group.make({
  GetProviderAuth,
  ListProviderAuth,
  UpdateProviderAuth,
  GetCloudUsage,
  GetProviderModelCatalog,
  RefreshModelCatalog,
  GetModelSlots,
  AssignSlot,
  ClearSlot,
  SetModelFavorite,
})
