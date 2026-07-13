import type {
  AgentCommunicationMessage,
  AssistantMessage,
  ContextUsageDisplay,
  DisplayAgent,
  DisplayActor,
  DisplayActorWork,
  DisplayActivity,
  DisplayMessage,
  DisplayState,
  DisplayTimeline,
  ErrorDisplayMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  GoalStatusMessage,
  InterruptedMessage,
  QueuedUserMessage,
  StatusIndicatorMessage,
  TaskAssignee,
  TaskDisplayRow,
  TaskWorkerState,
  ThinkingMessage,
  TimelineActivity,
  ToolMessage,
  UserMessage,
  WorkerFinishedMessage,
  WorkerKilledMessage,
  WorkerResumedMessage,
  WorkerUserKilledMessage,
  ForkResultMessage,
} from '@magnitudedev/protocol'
import { Addressed } from '@magnitudedev/event-core'
import { Schema } from 'effect'
import type { PendingInboundCommunication } from '../projections/turn'
import { PendingInboundCommunicationSchema } from '../projections/turn'

export type {
  AgentCommunicationMessage,
  AssistantMessage,
  ContextUsageDisplay,
  DisplayAgent,
  DisplayActor,
  DisplayActorWork,
  DisplayMessage,
  DisplayState,
  DisplayTimeline,
  ErrorDisplayMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  GoalStatusMessage,
  InterruptedMessage,
  QueuedUserMessage,
  StatusIndicatorMessage,
  TaskAssignee,
  TaskDisplayRow,
  TaskWorkerState,
  ThinkingMessage,
  DisplayActivity,
  TimelineActivity,
  ToolMessage,
  UserMessage,
  WorkerFinishedMessage,
  WorkerKilledMessage,
  WorkerResumedMessage,
  WorkerUserKilledMessage,
  ForkResultMessage,
}

export type UserMessageDisplay = UserMessage
export type QueuedUserMessageDisplay = QueuedUserMessage
export type AssistantMessageDisplay = AssistantMessage

export type PendingInboundCommunicationDisplay = PendingInboundCommunication

export const DisplayTimelineStateSchema = Schema.Struct({
  mode: Schema.Literal('idle', 'streaming'),
  messages: Addressed.AddressedSequenceIndexSchema,
  streamingMessageId: Schema.NullOr(Schema.String),
  _currentTurnId: Schema.NullOr(Schema.String),
  _pendingInboundCommunications: Schema.Array(PendingInboundCommunicationSchema),
  _queuedUserMessageCount: Schema.optionalWith(Schema.Number, { default: () => 0 }),
  _thinkingMessageId: Schema.NullOr(Schema.String),
  _activeToolCallIds: Schema.optionalWith(Schema.Array(Schema.String), { default: () => [] }),
  _communicationMessageIdsByStreamId: Schema.optionalWith(Schema.Record({
    key: Schema.String,
    value: Schema.String,
  }), { default: () => ({}) }),
  _forkActivityMessageIdsByForkId: Schema.optionalWith(Schema.Record({
    key: Schema.String,
    value: Schema.Array(Schema.String),
  }), { default: () => ({}) }),
})
export type DisplayTimelineState = typeof DisplayTimelineStateSchema.Type

export type StatusBarDecorator = 'spinner'
export type StatusToolKey = 'messageAdvisor' | 'messageWorker' | 'spawnWorker'
export type StatusBarActivity = DisplayActivity
