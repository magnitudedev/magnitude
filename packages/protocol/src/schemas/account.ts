import { Schema } from "effect"

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
  providerId: Schema.String,
  providerModelId: Schema.String,
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
  providerId: Schema.optional(Schema.String),
  providerModelId: Schema.optional(Schema.String),
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
  id: Schema.String,
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
  providerId: Schema.String,
  providerModelId: Schema.String,
  modelFamilyId: Schema.optional(Schema.String),
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

/** Provider/model discovery state. Slot selection is intentionally separate. */
export const ModelCatalogSchema = Schema.Struct({
  revision: Schema.NonNegativeInt,
  refreshing: Schema.Boolean,
  models: Schema.Array(ModelSummarySchema),
  providers: Schema.Array(ProviderInfoSchema),
})
export type ModelCatalog = Schema.Schema.Type<typeof ModelCatalogSchema>

/** Durable slot configuration plus its authoritative resolved projection. */
export const ModelSlotsSchema = Schema.Struct({
  revision: Schema.NonNegativeInt,
  profiles: SlotProfiles,
  config: ModelConfigResponseSchema,
})
export type ModelSlots = Schema.Schema.Type<typeof ModelSlotsSchema>

export const ModelResourceInvalidationSchema = Schema.TaggedStruct("changed", {
  revision: Schema.NonNegativeInt,
})
export type ModelResourceInvalidation = Schema.Schema.Type<typeof ModelResourceInvalidationSchema>

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
