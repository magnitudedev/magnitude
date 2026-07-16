import { Schema } from "effect"
import { ModelFamilyIdSchema, ProviderIdSchema, ProviderModelIdSchema } from "@magnitudedev/ai"
import { FSM } from "@magnitudedev/utils"
import { MirroredSnapshotSchema } from "./mirrored-resource"

const { defineFSM } = FSM

export const RoleId = Schema.Literal(
  "leader",
  "scout",
  "architect",
  "engineer",
  "critic",
  "scientist",
  "artisan",
  "advisor"
)
export type RoleId = Schema.Schema.Type<typeof RoleId>

export const UsagePeriod = Schema.Literal("24h", "3d", "7d", "14d", "30d", "all")
export type UsagePeriod = Schema.Schema.Type<typeof UsagePeriod>

// ---------------------------------------------------------------------------
// Slot-based model configuration
// ---------------------------------------------------------------------------

export const SlotId = Schema.Literal("primary", "secondary")
export type SlotId = Schema.Schema.Type<typeof SlotId>

/** Effective slot config (read-only, sent to client for display). */
export const SlotProfile = Schema.Struct({
  slotId: SlotId,
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  modelDisplayName: Schema.String,
  contextWindow: Schema.Number,
  maxOutputTokens: Schema.Number,
  capabilities: Schema.Struct({ vision: Schema.Boolean }),
  reasoningEffort: Schema.String,            // plain string — values come from ProviderModel.reasoningEfforts
  isUserOverride: Schema.Boolean,
})
export type SlotProfile = Schema.Schema.Type<typeof SlotProfile>

export const SlotProfiles = Schema.partial(
  Schema.Record({ key: SlotId, value: SlotProfile })
)
export type SlotProfiles = Schema.Schema.Type<typeof SlotProfiles>

/** Persisted per-slot user config (providerId/providerModelId/reasoningEffort overrides). */
export const SlotModelConfigSchema = Schema.Struct({
  providerId: Schema.optional(ProviderIdSchema),
  providerModelId: Schema.optional(ProviderModelIdSchema),
  reasoningEffort: Schema.optional(Schema.String),
})
export type SlotModelConfig = Schema.Schema.Type<typeof SlotModelConfigSchema>

export const ModelConfigResponseSchema = Schema.Struct({
  slots: Schema.partial(Schema.Record({ key: SlotId, value: SlotModelConfigSchema })),
  localSlotIntent: Schema.partial(Schema.Record({ key: SlotId, value: Schema.Literal("local", "cloud") })),
})
export type ModelConfigResponse = Schema.Schema.Type<typeof ModelConfigResponseSchema>

// ---------------------------------------------------------------------------
// Provider info
// ---------------------------------------------------------------------------

export const AuthStatusSchema = Schema.Literal("authenticated", "no_auth_required", "not_configured")
export type AuthStatus = Schema.Schema.Type<typeof AuthStatusSchema>

export const ProviderInfoSchema = Schema.Struct({
  id: ProviderIdSchema,
  displayName: Schema.String,
  authStatus: AuthStatusSchema,
  status: Schema.optional(Schema.Literal("ok", "loading", "not_found", "error")),
  message: Schema.optional(Schema.String),
  hint: Schema.optional(Schema.String),
})
export type ProviderInfo = Schema.Schema.Type<typeof ProviderInfoSchema>

export const ProviderModelDisabledReason = Schema.Literal(
  "insufficient_resources",
  "provider_unavailable",
  "model_unavailable",
  "incompatible_runtime",
  "invalid_configuration",
)
export type ProviderModelDisabledReason = Schema.Schema.Type<typeof ProviderModelDisabledReason>

export const ProviderModelAvailability = Schema.Union(
  Schema.TaggedStruct("Available", {}),
  Schema.TaggedStruct("Disabled", { reason: ProviderModelDisabledReason }),
)
export type ProviderModelAvailability = Schema.Schema.Type<typeof ProviderModelAvailability>

// ---------------------------------------------------------------------------
// Model summary (provider catalog entry)
// ---------------------------------------------------------------------------

export const ModelSummarySchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  modelFamilyId: Schema.optional(ModelFamilyIdSchema),
  displayName: Schema.String,
  slots: Schema.optional(Schema.Array(SlotId)),
  contextWindow: Schema.Number,
  maxOutputTokens: Schema.Number,
  capabilities: Schema.Struct({ vision: Schema.optional(Schema.Boolean) }),
  availability: ProviderModelAvailability,
  reasoningEfforts: Schema.Array(Schema.String),
  pricing: Schema.optional(Schema.Struct({
    input: Schema.Number,
    output: Schema.Number,
    cachedInput: Schema.optional(Schema.Number),
  })),
})
export type ModelSummary = Schema.Schema.Type<typeof ModelSummarySchema>

export class ProviderCatalogStale extends Schema.TaggedClass<ProviderCatalogStale>()("stale", {
  providerId: ProviderIdSchema,
  message: Schema.String,
}) {}

export class ProviderCatalogUnavailable extends Schema.TaggedClass<ProviderCatalogUnavailable>()("unavailable", {
  providerId: ProviderIdSchema,
  message: Schema.String,
}) {}

export const ProviderCatalogFailureSchema = Schema.Union(ProviderCatalogStale, ProviderCatalogUnavailable)
export type ProviderCatalogFailure = typeof ProviderCatalogFailureSchema.Type

export class ModelSlotConfigurationUnavailable extends Schema.TaggedClass<ModelSlotConfigurationUnavailable>()("configuration_unavailable", {
  message: Schema.String,
}) {}

export const ModelSlotsFailureSchema = Schema.Union(
  ProviderCatalogFailureSchema,
  ModelSlotConfigurationUnavailable,
)
export type ModelSlotsFailure = typeof ModelSlotsFailureSchema.Type

const CatalogSnapshotFields = {
  models: Schema.Array(ModelSummarySchema),
  providers: Schema.Array(ProviderInfoSchema),
} as const

export class ModelCatalogLoading extends Schema.TaggedClass<ModelCatalogLoading>()("loading", {}) {}
export class ModelCatalogReady extends Schema.TaggedClass<ModelCatalogReady>()("ready", CatalogSnapshotFields) {}
export class ModelCatalogRefreshing extends Schema.TaggedClass<ModelCatalogRefreshing>()("refreshing", {
  ...CatalogSnapshotFields,
  failures: Schema.Array(ProviderCatalogFailureSchema),
}) {}
export class ModelCatalogDegraded extends Schema.TaggedClass<ModelCatalogDegraded>()("degraded", {
  ...CatalogSnapshotFields,
  failures: Schema.Array(ProviderCatalogFailureSchema),
}) {}
export class ModelCatalogUnavailable extends Schema.TaggedClass<ModelCatalogUnavailable>()("unavailable", {
  providers: Schema.Array(ProviderInfoSchema),
  failures: Schema.Array(ProviderCatalogUnavailable),
}) {}

export const ModelCatalogLifecycle = defineFSM(
  {
    loading: ModelCatalogLoading,
    ready: ModelCatalogReady,
    refreshing: ModelCatalogRefreshing,
    degraded: ModelCatalogDegraded,
    unavailable: ModelCatalogUnavailable,
  },
  {
    loading: ["ready", "degraded", "unavailable"],
    ready: ["refreshing"],
    refreshing: ["ready", "degraded", "unavailable"],
    degraded: ["refreshing"],
    unavailable: ["refreshing"],
  } as const,
)

export const ModelCatalogStateSchema = Schema.Union(
  ModelCatalogLoading,
  ModelCatalogReady,
  ModelCatalogRefreshing,
  ModelCatalogDegraded,
  ModelCatalogUnavailable,
)
export type ModelCatalogState = typeof ModelCatalogStateSchema.Type

/** Provider/model discovery state. Slot selection is intentionally separate. */
export const ModelCatalogSchema = MirroredSnapshotSchema(ModelCatalogStateSchema)
export type ModelCatalog = Schema.Schema.Type<typeof ModelCatalogSchema>

const SlotSnapshotFields = {
  profiles: SlotProfiles,
  config: ModelConfigResponseSchema,
} as const

export class ModelSlotsLoading extends Schema.TaggedClass<ModelSlotsLoading>()("loading", {}) {}
export class ModelSlotsReady extends Schema.TaggedClass<ModelSlotsReady>()("ready", SlotSnapshotFields) {}
export class ModelSlotsRefreshing extends Schema.TaggedClass<ModelSlotsRefreshing>()("refreshing", {
  ...SlotSnapshotFields,
  failures: Schema.Array(ModelSlotsFailureSchema),
}) {}
export class ModelSlotsDegraded extends Schema.TaggedClass<ModelSlotsDegraded>()("degraded", {
  ...SlotSnapshotFields,
  failures: Schema.Array(ModelSlotsFailureSchema),
}) {}
export class ModelSlotsUnavailable extends Schema.TaggedClass<ModelSlotsUnavailable>()("unavailable", {
  config: ModelConfigResponseSchema,
  failures: Schema.Array(ModelSlotsFailureSchema),
}) {}

export const ModelSlotsLifecycle = defineFSM(
  {
    loading: ModelSlotsLoading,
    ready: ModelSlotsReady,
    refreshing: ModelSlotsRefreshing,
    degraded: ModelSlotsDegraded,
    unavailable: ModelSlotsUnavailable,
  },
  {
    loading: ["ready", "degraded", "unavailable"],
    ready: ["refreshing"],
    refreshing: ["ready", "degraded", "unavailable"],
    degraded: ["refreshing"],
    unavailable: ["refreshing"],
  } as const,
)

export const ModelSlotsStateSchema = Schema.Union(
  ModelSlotsLoading,
  ModelSlotsReady,
  ModelSlotsRefreshing,
  ModelSlotsDegraded,
  ModelSlotsUnavailable,
)
export type ModelSlotsState = typeof ModelSlotsStateSchema.Type

/** Durable slot configuration plus its authoritative resolved projection. */
export const ModelSlotsSchema = MirroredSnapshotSchema(ModelSlotsStateSchema)
export type ModelSlots = Schema.Schema.Type<typeof ModelSlotsSchema>

// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export const ApiKeyAuthSchema = Schema.Struct({
  type: Schema.Literal("api"),
  key: Schema.String,
})

export const EndpointAuthSchema = Schema.Struct({
  type: Schema.Literal("endpoint"),
  endpoint: Schema.String,
  apiKey: Schema.optional(Schema.String),
})

export const ProviderAuthSchema = Schema.Union(ApiKeyAuthSchema, EndpointAuthSchema)
export type ProviderAuth = Schema.Schema.Type<typeof ProviderAuthSchema>

// ---------------------------------------------------------------------------
// Balance
// ---------------------------------------------------------------------------

const UsageTotals = Schema.Struct({
  requestCount: Schema.Number,
  costCents: Schema.Number,
  inputTokens: Schema.Number,
  outputTokens: Schema.Number
})

export const BalanceResponse = Schema.Struct({
  data: Schema.Struct({
    meta: Schema.Struct({
      generatedAt: Schema.String,
      autumnConfigured: Schema.Boolean
    }),
    balance: Schema.Struct({
      cents: Schema.Number
    }),
    autoReload: Schema.Struct({
      enabled: Schema.Boolean,
      thresholdCents: Schema.Number,
      amountCents: Schema.Number,
      lastFailure: Schema.NullOr(Schema.Struct({
        reason: Schema.String,
        at: Schema.NullOr(Schema.String)
      }))
    }),
    hasPaymentMethod: Schema.Boolean,
    recentTopups: Schema.Array(Schema.Struct({
      at: Schema.NullOr(Schema.String),
      chargedCents: Schema.Number,
      invoiceUrl: Schema.NullOr(Schema.String),
      status: Schema.NullOr(Schema.String)
    })),
    usage: Schema.Struct({
      period: UsagePeriod,
      rangeStart: Schema.String,
      rangeEnd: Schema.String,
      totals: UsageTotals,
      byModel: Schema.Array(Schema.Struct({
        model: Schema.String,
        requestCount: Schema.Number,
        costCents: Schema.Number,
        inputTokens: Schema.Number,
        outputTokens: Schema.Number
      })),
      dailyTokens: Schema.Array(Schema.Struct({
        date: Schema.String,
        inputTokens: Schema.Number,
        outputTokens: Schema.Number,
        topModel: Schema.NullOr(Schema.String)
      }))
    })
  })
})
export type BalanceResponse = Schema.Schema.Type<typeof BalanceResponse>
