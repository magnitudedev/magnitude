/**
 * ATIF (Agent Trajectory Interchange Format) v1.7 types
 * Native Magnitude projection types mirroring the Harbor ATIF spec.
 */

import { Option, Schema } from 'effect'
import { JsonValueSchema } from '@magnitudedev/ai'
import { ROLE_IDS } from '../../agents/role-validation'

// =============================================================================
// Content / Message Types
// =============================================================================

const JsonRecordSchema = Schema.Record({ key: Schema.String, value: JsonValueSchema })

export const AtifTextPartSchema = Schema.Struct({
  type: Schema.Literal('text'),
  text: Schema.String,
})
export type AtifTextPart = typeof AtifTextPartSchema.Type

export const AtifImageSourceSchema = Schema.Struct({
  media_type: Schema.Literal('image/jpeg', 'image/png', 'image/gif', 'image/webp'),
  path: Schema.String,
})
export type AtifImageSource = typeof AtifImageSourceSchema.Type

export const AtifImagePartSchema = Schema.Struct({
  type: Schema.Literal('image'),
  source: AtifImageSourceSchema,
})
export type AtifImagePart = typeof AtifImagePartSchema.Type

export const AtifContentPartSchema = Schema.Union(AtifTextPartSchema, AtifImagePartSchema)
export type AtifContentPart = typeof AtifContentPartSchema.Type

export const AtifMessageSchema = Schema.Union(Schema.String, Schema.Array(AtifContentPartSchema))
export type AtifMessage = typeof AtifMessageSchema.Type

// =============================================================================
// Tool Call
// =============================================================================

export const AtifToolCallSchema = Schema.Struct({
  tool_call_id: Schema.String,
  function_name: Schema.String,
  arguments: JsonRecordSchema,
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifToolCall = typeof AtifToolCallSchema.Type

// =============================================================================
// Observation
// =============================================================================

export const AtifSubagentTrajectoryRefSchema = Schema.Struct({
  trajectory_id: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  trajectory_path: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  session_id: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifSubagentTrajectoryRef = typeof AtifSubagentTrajectoryRefSchema.Type

export const AtifObservationResultSchema = Schema.Struct({
  source_call_id: Schema.optionalWith(Schema.NullOr(Schema.String), { as: 'Option', exact: true }),
  content: Schema.optionalWith(AtifMessageSchema, { as: 'Option', exact: true }),
  subagent_trajectory_ref: Schema.optionalWith(Schema.Array(AtifSubagentTrajectoryRefSchema), { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifObservationResult = typeof AtifObservationResultSchema.Type

export const AtifObservationSchema = Schema.Struct({
  results: Schema.Array(AtifObservationResultSchema),
})
export type AtifObservation = typeof AtifObservationSchema.Type

// =============================================================================
// Metrics
// =============================================================================

export const AtifMetricsSchema = Schema.Struct({
  prompt_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  completion_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  cached_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  cost_usd: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  prompt_token_ids: Schema.optionalWith(Schema.Array(Schema.Number), { as: 'Option', exact: true }),
  completion_token_ids: Schema.optionalWith(Schema.Array(Schema.Number), { as: 'Option', exact: true }),
  logprobs: Schema.optionalWith(Schema.Array(Schema.Number), { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifMetrics = typeof AtifMetricsSchema.Type

// =============================================================================
// Step
// =============================================================================

export const AtifStepSourceSchema = Schema.Literal('system', 'user', 'agent')
export type AtifStepSource = typeof AtifStepSourceSchema.Type

export const AtifStepSchema = Schema.Struct({
  step_id: Schema.Number,
  timestamp: Schema.optionalWith(Schema.NullOr(Schema.String), { as: 'Option', exact: true }),
  source: AtifStepSourceSchema,
  model_name: Schema.optionalWith(Schema.NullOr(Schema.String), { as: 'Option', exact: true }),
  reasoning_effort: Schema.optionalWith(Schema.NullOr(Schema.Union(Schema.String, Schema.Number)), { as: 'Option', exact: true }),
  message: AtifMessageSchema,
  reasoning_content: Schema.optionalWith(Schema.NullOr(Schema.String), { as: 'Option', exact: true }),
  tool_calls: Schema.optionalWith(Schema.Array(AtifToolCallSchema), { as: 'Option', exact: true }),
  observation: Schema.optionalWith(AtifObservationSchema, { as: 'Option', exact: true }),
  metrics: Schema.optionalWith(AtifMetricsSchema, { as: 'Option', exact: true }),
  is_copied_context: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
  llm_call_count: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifStep = typeof AtifStepSchema.Type
export type AtifStepDraft = Omit<AtifStep, 'step_id'>

// =============================================================================
// Agent Info
// =============================================================================

export const AtifAgentSchema = Schema.Struct({
  name: Schema.String,
  version: Schema.String,
  model_name: Schema.optionalWith(Schema.NullOr(Schema.String), { as: 'Option', exact: true }),
  tool_definitions: Schema.optionalWith(Schema.Array(JsonRecordSchema), { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifAgent = typeof AtifAgentSchema.Type

// =============================================================================
// Final Metrics
// =============================================================================

export const AtifFinalMetricsSchema = Schema.Struct({
  total_prompt_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  total_completion_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  total_cached_tokens: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  total_cost_usd: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  total_steps: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
})
export type AtifFinalMetrics = typeof AtifFinalMetricsSchema.Type

// =============================================================================
// Trajectory
// =============================================================================

const AtifTrajectoryFields = {
  schema_version: Schema.Literal('ATIF-v1.7'),
  session_id: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  trajectory_id: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  agent: AtifAgentSchema,
  steps: Schema.Array(AtifStepSchema),
  notes: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  final_metrics: Schema.optionalWith(AtifFinalMetricsSchema, { as: 'Option', exact: true }),
  continued_trajectory_ref: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  extra: Schema.optionalWith(JsonRecordSchema, { as: 'Option', exact: true }),
}

const AtifTrajectoryBaseSchema = Schema.Struct(AtifTrajectoryFields)
type AtifTrajectoryBase = Schema.Schema.Type<typeof AtifTrajectoryBaseSchema>
type AtifTrajectoryBaseEncoded = Schema.Schema.Encoded<typeof AtifTrajectoryBaseSchema>

export interface AtifTrajectory extends AtifTrajectoryBase {
  readonly subagent_trajectories: Option.Option<ReadonlyArray<AtifTrajectory>>
}

interface AtifTrajectoryEncoded extends AtifTrajectoryBaseEncoded {
  readonly subagent_trajectories?: ReadonlyArray<AtifTrajectoryEncoded>
}

export const AtifTrajectorySchema: Schema.Schema<AtifTrajectory, AtifTrajectoryEncoded> = Schema.Struct({
  ...AtifTrajectoryFields,
  subagent_trajectories: Schema.optionalWith(
    Schema.Array(Schema.suspend(
      (): Schema.Schema<AtifTrajectory, AtifTrajectoryEncoded> => AtifTrajectorySchema,
    )),
    { as: 'Option', exact: true },
  ),
})

// =============================================================================
// Projection State
// =============================================================================

export const PendingToolCallSchema = Schema.Struct({
  toolCallId: Schema.String,
  function_name: Schema.String,
  arguments: JsonRecordSchema,
})
export type PendingToolCall = typeof PendingToolCallSchema.Type

/** In-progress agent turn keyed by Magnitude turnId, not by ATIF step_id. */
export const ActiveAtifTurnSchema = Schema.Struct({
  turnId: Schema.String,
  chainId: Schema.String,
  forkId: Schema.NullOr(Schema.String),
  source: Schema.Literal('agent'),
  timestamp: Schema.NullOr(Schema.String),
  model_name: Schema.NullOr(Schema.String),
  message: Schema.String,
  reasoning_content: Schema.String,
  tool_calls: Schema.Array(AtifToolCallSchema),
  observation_results: Schema.Array(AtifObservationResultSchema),
  pendingToolCalls: Schema.ReadonlyMap({ key: Schema.String, value: PendingToolCallSchema }),
  metrics: Schema.NullOr(AtifMetricsSchema),
  llm_call_count: Schema.Number,
})
export type ActiveAtifTurn = typeof ActiveAtifTurnSchema.Type

export const TokenAccumulatorSchema = Schema.Struct({
  promptTokens: Schema.Number,
  completionTokens: Schema.Number,
  cachedTokens: Schema.Number,
  costUsd: Schema.Number,
})
export type TokenAccumulator = typeof TokenAccumulatorSchema.Type

const RoleIdSchema = Schema.Literal(...ROLE_IDS)

export const AtifForkStateSchema = Schema.Struct({
  forkId: Schema.NullOr(Schema.String),
  agentName: Schema.String,
  agentRole: Schema.NullOr(RoleIdSchema),
  modelId: Schema.NullOr(Schema.String),
  steps: Schema.Array(AtifStepSchema),
  activeTurns: Schema.ReadonlyMap({ key: Schema.String, value: ActiveAtifTurnSchema }),
  compactionBoundaryIndex: Schema.NullOr(Schema.Number),
  tokenAccumulator: TokenAccumulatorSchema,
})
export type AtifForkState = typeof AtifForkStateSchema.Type

export const AtifProjectionStateSchema = Schema.Struct({
  forks: Schema.ReadonlyMap({ key: Schema.NullOr(Schema.String), value: AtifForkStateSchema }),
})
export type AtifProjectionState = typeof AtifProjectionStateSchema.Type
