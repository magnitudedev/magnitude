import { Schema } from "effect"
import {
  ModelFamilyIdSchema,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
} from "@magnitudedev/ai"
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
  reasoningEffort: ReasoningEffortSchema,
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
  reasoningEffort: Schema.optional(ReasoningEffortSchema),
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
  "installation_unavailable",
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
  defaultReasoningEffort: ReasoningEffortSchema,
  properties: Schema.Struct({
    vision: VisionProperty.Schema,
    reasoning: ReasoningProperty.Schema,
  }),
  availability: ProviderModelAvailability,
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

export const SlotSelectionSourceSchema = Schema.Literal("automatic", "user")
export const SlotSelectionSchema = Schema.Struct({
  providerId: ProviderIdSchema,
  providerModelId: ProviderModelIdSchema,
  reasoningEffort: ReasoningEffortSchema,
})
export class SlotUnassigned extends Schema.TaggedClass<SlotUnassigned>()("Unassigned", {
  slotId: SlotId,
  reason: Schema.Literal("no_candidate", "provider_unavailable"),
}) {}
export class SlotPending extends Schema.TaggedClass<SlotPending>()("Pending", {
  slotId: SlotId,
  selection: SlotSelectionSchema,
  source: SlotSelectionSourceSchema,
  waitingFor: Schema.Array(Schema.Literal("vision", "reasoning")).pipe(Schema.minItems(1)),
}) {}
export class SlotReady extends Schema.TaggedClass<SlotReady>()("Ready", {
  slotId: SlotId,
  selection: SlotSelectionSchema,
  source: SlotSelectionSourceSchema,
  modelDisplayName: Schema.String,
  contextWindow: Schema.Number.pipe(Schema.int(), Schema.positive()),
  maxOutputTokens: Schema.Number.pipe(Schema.int(), Schema.positive()),
}) {}
export class SlotBlocked extends Schema.TaggedClass<SlotBlocked>()("Blocked", {
  slotId: SlotId,
  selection: SlotSelectionSchema,
  source: SlotSelectionSourceSchema,
  reason: Schema.Literal("model_unavailable", "model_removed", "installation_unavailable", "incompatible_runtime", "invalid_configuration", "property_discovery_failed"),
}) {}
export const SlotStateSchema = Schema.Union(SlotUnassigned, SlotPending, SlotReady, SlotBlocked)
export type SlotState = typeof SlotStateSchema.Type
export const SlotStatesSchema = Schema.Record({ key: SlotId, value: SlotStateSchema })
export type SlotStates = typeof SlotStatesSchema.Type

const SlotSnapshotFields = {
  slots: SlotStatesSchema,
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
  slots: SlotStatesSchema,
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
// Cloud subscription and usage limits
// ---------------------------------------------------------------------------

const UsageTotals = Schema.Struct({
  requestCount: Schema.Number,
  costCents: Schema.Number,
  inputTokens: Schema.Number,
  outputTokens: Schema.Number
})

export const CloudUsageResponse = Schema.Struct({
  data: Schema.Struct({
    meta: Schema.Struct({
      generatedAt: Schema.String,
      autumnConfigured: Schema.Boolean
    }),
    subscription: Schema.Struct({
      status: Schema.Literal("active", "not_subscribed"),
      plan: Schema.Struct({
        id: Schema.Literal("pro"),
        label: Schema.Literal("Pro"),
        priceCents: Schema.Number,
      }),
    }),
    usageWindows: Schema.partial(Schema.Record({
      key: Schema.Literal("five_hour", "weekly", "monthly"),
      value: Schema.Struct({
        limitCents: Schema.Number,
        usedCents: Schema.Number,
        remainingCents: Schema.Number,
        windowStart: Schema.String,
        windowEnd: Schema.String,
        remainingMs: Schema.Number,
      }),
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
export type CloudUsageResponse = Schema.Schema.Type<typeof CloudUsageResponse>
