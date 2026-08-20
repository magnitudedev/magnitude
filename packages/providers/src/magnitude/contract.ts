/**
 * Magnitude provider contract types.
 *
 * These types define the Magnitude-specific extension of ProviderModel
 * and the call options for the Magnitude provider.
 */

import type { RoleId } from "./roles"
import type { SlotId } from "@magnitudedev/roles"
import type { ProviderModel, ReasoningEffort, ModelPricingInfo } from "@magnitudedev/ai"
import { Schema } from "effect"
import { ProviderModelIdSchema } from "@magnitudedev/ai"

export type { ReasoningEffort, ModelPricingInfo } from "@magnitudedev/ai"
export type { ToolChoice } from "@magnitudedev/ai"
export type { ProviderModelCapabilities as ModelCapabilities } from "@magnitudedev/ai"

/**
 * A model in the Magnitude provider's catalog.
 * Extends ProviderModel with Magnitude-specific fields.
 */
export interface MagnitudeModelInfo extends ProviderModel {
  readonly object: "model"
  readonly owned_by: string
  readonly roles: readonly RoleId[]
  readonly slots: readonly SlotId[]
  readonly type?: "utility"
}

const MagnitudeRoleIdSchema: Schema.Schema<RoleId> = Schema.Literal(
  "leader",
  "scout",
  "architect",
  "engineer",
  "critic",
  "scientist",
  "artisan",
  "advisor",
)

const MagnitudeSlotIdSchema: Schema.Schema<SlotId> = Schema.Literal("primary", "secondary")

/** Validated raw model shape returned by Magnitude model-list endpoints. */
export const MagnitudeRawModelSchema = Schema.Struct({
  id: ProviderModelIdSchema,
  object: Schema.Literal("model"),
  owned_by: Schema.String,
  displayName: Schema.String,
  roles: Schema.Array(MagnitudeRoleIdSchema),
  slots: Schema.Array(MagnitudeSlotIdSchema),
  tiers: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  type: Schema.optionalWith(Schema.Literal("utility"), { as: "Option", exact: true }),
  contextWindow: Schema.Number,
  maxOutputTokens: Schema.Number,
  capabilities: Schema.optionalWith(Schema.Struct({
    vision: Schema.Boolean,
    structuredOutput: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
  pricing: Schema.optionalWith(Schema.Struct({
    input: Schema.Number,
    output: Schema.Number,
    cached_input: Schema.NullOr(Schema.Number),
  }), { as: "Option", exact: true }),
})
export type MagnitudeRawModel = Schema.Schema.Type<typeof MagnitudeRawModelSchema>

export const MagnitudeModelListResponseSchema = Schema.Struct({
  object: Schema.Literal("list"),
  data: Schema.Array(MagnitudeRawModelSchema),
})
export type ModelListResponse = Schema.Schema.Type<typeof MagnitudeModelListResponseSchema>

export const TurnConstraintsSchema = Schema.Struct({
  message: Schema.optionalWith(
    Schema.Literal("force", "allow", "forbid"),
    { as: "Option", exact: true },
  ),
})

export type TurnConstraints = Schema.Schema.Type<typeof TurnConstraintsSchema>

export const MagnitudeAdditionalOptionsSchema = Schema.Struct({
  traits: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  forceTrait: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  turn_constraints: Schema.optionalWith(TurnConstraintsSchema, { as: "Option", exact: true }),
  session_id: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  agent_id: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  include_raw: Schema.optionalWith(Schema.Boolean, { as: "Option", exact: true }),
  prefer_provider: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
})

export type MagnitudeAdditionalOptions = Schema.Schema.Type<typeof MagnitudeAdditionalOptionsSchema>

export type {
  BillingWindowBudget,
  BillingWindowName,
  MagnitudeApiError,
  MagnitudeErrorCode,
  MagnitudeErrorDetails,
  MagnitudeErrorType,
  ProSubscriptionStatus,
  SubscriptionRequiredDetails,
  UsageLimitDetails,
} from "./generated-contract/errors"
