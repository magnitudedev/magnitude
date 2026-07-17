import { AssistantMessageSchema } from '@magnitudedev/ai'
import { ToolResultEntrySchema } from '@magnitudedev/harness'
import { Schema, Option } from 'effect'
import { TimelineEntrySchema } from './inbox/types'
import { ContextPartSchema } from '../content'

// ---------------------------------------------------------------------------
// CompletedTurn / TurnFeedback
// ---------------------------------------------------------------------------

export const TurnFeedbackSchema = Schema.Union(
  Schema.Struct({ kind: Schema.Literal('message_ack'), destination: Schema.Literal('coordinator'), chars: Schema.Number }),
  Schema.Struct({ kind: Schema.Literal('error'), message: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('overthinking'), message: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('interrupted') }),
)
export type TurnFeedback = typeof TurnFeedbackSchema.Type

export const CompletedTurnSchema = Schema.Struct({
  turnId: Schema.String,
  assistant: AssistantMessageSchema,
  toolResults: Schema.Array(ToolResultEntrySchema),
  feedback: Schema.Array(TurnFeedbackSchema),
  clean: Schema.Boolean,
})
export type CompletedTurn = typeof CompletedTurnSchema.Type

// ---------------------------------------------------------------------------
// Window entries
// ---------------------------------------------------------------------------

export type WindowEntrySource = 'user' | 'agent' | 'system'

export const WindowEntrySchema = Schema.Union(
  Schema.Struct({ type: Schema.Literal('session_context'), source: Schema.Literal('system'), content: Schema.Array(ContextPartSchema), estimatedTokens: Schema.Number }),
  Schema.Struct({ type: Schema.Literal('assistant_turn'), source: Schema.Literal('agent'), turn: CompletedTurnSchema, strategyId: Schema.Literal('native'), estimatedTokens: Schema.Number }),
  Schema.Struct({ type: Schema.Literal('compacted'), source: Schema.Literal('system'), content: Schema.Array(ContextPartSchema), estimatedTokens: Schema.Number }),
  Schema.Struct({ type: Schema.Literal('fork_context'), source: Schema.Literal('system'), content: Schema.Array(ContextPartSchema), estimatedTokens: Schema.Number }),
  Schema.Struct({ type: Schema.Literal('goal_injection'), source: Schema.Literal('system'), content: Schema.Array(ContextPartSchema), estimatedTokens: Schema.Number }),
  Schema.Struct({ type: Schema.Literal('context'), source: Schema.Literal('system'), timeline: Schema.Array(TimelineEntrySchema), estimatedTokens: Schema.Number }),
  Schema.Struct({
    type: Schema.Literal('observer_turn'),
    source: Schema.Literal('system'),
    observerTurnId: Schema.String,
    estimatedTokens: Schema.Number,
    justification: Schema.NullOr(Schema.Literal('difficulty', 'churn', 'frustration')),
    escalate: Schema.Boolean,
    reasoning: Schema.String,
  }),
  Schema.Struct({ type: Schema.Literal('advisor_response'), source: Schema.Literal('system'), mode: Schema.Literal('advice'), content: Schema.String, estimatedTokens: Schema.Number }),
)

export type WindowEntry = typeof WindowEntrySchema.Type

export const QueuedTimelineEntrySchema = Schema.Struct({
  timestamp: Schema.Number,
  seq: Schema.Number,
  entry: TimelineEntrySchema,
  coalesceKey: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
})
export type QueuedTimelineEntry = typeof QueuedTimelineEntrySchema.Type

export const ForkWindowStateSchema = Schema.Struct({
  messages: Schema.Array(WindowEntrySchema),
  queuedTimeline: Schema.Array(QueuedTimelineEntrySchema),
  currentTurnId: Schema.NullOr(Schema.String),
  currentChainId: Schema.NullOr(Schema.String),
  nextQueueSeq: Schema.Number,
  _activeMessageIsCoordinator: Schema.Boolean,
  _coordinatorChars: Schema.Number,
  tokenEstimate: Schema.Number,
  messageTokens: Schema.Number,
  systemPromptTokens: Schema.Number,
  lastAnchoredTotal: Schema.NullOr(Schema.Number),
  lastAnchoredMessageTokens: Schema.NullOr(Schema.Number),
  autopilotEnabled: Schema.Boolean,
  consumerAutopilotKnowledge: Schema.Struct({
    advisor: Schema.NullOr(Schema.Boolean),
    leader: Schema.NullOr(Schema.Boolean),
  }),
})
export type ForkWindowState = typeof ForkWindowStateSchema.Type
