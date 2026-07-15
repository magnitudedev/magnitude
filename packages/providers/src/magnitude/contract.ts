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

export type { ReasoningEffort, ModelPricingInfo } from "@magnitudedev/ai"
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
  id: Schema.String,
  object: Schema.Literal("model"),
  owned_by: Schema.String,
  displayName: Schema.String,
  roles: Schema.Array(MagnitudeRoleIdSchema),
  slots: Schema.Array(MagnitudeSlotIdSchema),
  tiers: Schema.optional(Schema.Array(Schema.String)),
  type: Schema.optional(Schema.Literal("utility")),
  contextWindow: Schema.Number,
  maxOutputTokens: Schema.Number,
  capabilities: Schema.optional(Schema.Struct({ vision: Schema.Boolean })),
  pricing: Schema.optional(Schema.Struct({
    input: Schema.Number,
    output: Schema.Number,
    cached_input: Schema.NullOr(Schema.Number),
  })),
  reasoningEfforts: Schema.optional(Schema.Array(Schema.String)),
})
export type MagnitudeRawModel = Schema.Schema.Type<typeof MagnitudeRawModelSchema>

export const MagnitudeModelListResponseSchema = Schema.Struct({
  object: Schema.Literal("list"),
  data: Schema.Array(MagnitudeRawModelSchema),
})
export type ModelListResponse = Schema.Schema.Type<typeof MagnitudeModelListResponseSchema>

export type ToolChoice =
  | "none"
  | "auto"
  | "required"
  | NamedFunctionToolChoice
  | AllowedToolsToolChoice
  | GrammarToolChoice

export type NamedFunctionToolChoice = {
  type: "function"
  function: { name: string }
}

export type AllowedToolsToolChoice = {
  type: "allowed_tools"
  allowed_tools: {
    mode: "auto" | "required"
    tools: Array<{ type: "function"; function: { name: string } }>
  }
}

export type GrammarToolChoice = {
  type: "grammar"
  grammar: string
}

export type TurnConstraintMessage = "force" | "allow" | "forbid"

export type TurnConstraints = {
  message?: TurnConstraintMessage
}

export type MagnitudeAdditionalOptions = {
  traits?: string[]
  forceTrait?: string
  turn_constraints?: TurnConstraints
  session_id?: string
  agent_id?: string
  include_raw?: boolean
  prefer_provider?: string
}

export interface MagnitudeApiError {
  readonly error: {
    readonly message: string
    readonly type: MagnitudeErrorType
    readonly code: MagnitudeErrorCode
    readonly param: string | null
    readonly details?: MagnitudeErrorDetails
  }
}

export type MagnitudeErrorType =
  | "invalid_request_error"
  | "authentication_error"
  | "insufficient_quota"
  | "rate_limit_error"
  | "server_error"
  | "service_unavailable"

export type MagnitudeErrorCode =
  | "invalid_api_key"
  | "invalid_body"
  | "unsupported_field"
  | "unsupported_n"
  | "invalid_image_url"
  | "invalid_multimodal_role"
  | "model_not_found"
  | "model_not_multimodal"
  | "model_not_grammar_compatible"
  | "insufficient_credits"
  | "provider_rate_limited"
  | "internal_server_error"
  | "provider_error"
  | "invariant_violation"
  | "upstream_unavailable"
  | "stream_interrupted"

export type MagnitudeErrorDetails = InsufficientCreditsDetails

export interface InsufficientCreditsDetails {
  readonly category: "insufficient_credits"
  readonly balanceCents: number
}
