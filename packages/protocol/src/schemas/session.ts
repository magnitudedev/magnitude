import { Schema } from "effect"
import { RawImageAttachment, RawMentionOccurrence } from "./attachments"
export {
  RawClipboardImageAttachment,
  DisplayAttachment,
  RawFileImageAttachment,
  RawImageAttachment,
  ImageAttachment,
  ImageMediaType,
  MentionAttachment,
  MentionDirectoryAttachment,
  MentionFileAttachment,
  MentionFileRangeAttachment,
  MessageAttachment,
  RawMentionOccurrence,
} from "./attachments"

export const CreateSessionInitialMessage = Schema.TaggedStruct("message", {
  messageId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  content: Schema.String,
  visibleMessage: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  taskMode: Schema.Boolean,
  imageAttachments: Schema.Array(RawImageAttachment),
  mentions: Schema.Array(RawMentionOccurrence),
})
export type CreateSessionInitialMessage = Schema.Schema.Type<typeof CreateSessionInitialMessage>

export const CreateSessionInitialGoal = Schema.TaggedStruct("goal", {
  objective: Schema.String
})
export type CreateSessionInitialGoal = Schema.Schema.Type<typeof CreateSessionInitialGoal>

export const CreateSessionInitial = Schema.Union(
  CreateSessionInitialMessage,
  CreateSessionInitialGoal
)
export type CreateSessionInitial = Schema.Schema.Type<typeof CreateSessionInitial>

export const SessionOptions = Schema.Struct({
  disableShellSafeguards: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  disableCwdSafeguards: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  atifPath: Schema.optional(Schema.String),
  solo: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  systemPromptOverride: Schema.optional(Schema.String),
  headless: Schema.optionalWith(Schema.Boolean, { default: () => false })
})
export type SessionOptions = Schema.Schema.Type<typeof SessionOptions>

export const PreloadSessionResult = Schema.Struct({
  sessionId: Schema.String,
})
export type PreloadSessionResult = Schema.Schema.Type<typeof PreloadSessionResult>

export const SessionMetadata = Schema.Struct({
  sessionId: Schema.String,
  title: Schema.Union(Schema.String, Schema.Null),
  cwd: Schema.String,
  createdAt: Schema.Number,
  updatedAt: Schema.Number,
  messageCount: Schema.Number,
  lastMessage: Schema.Union(Schema.String, Schema.Null)
})
export type SessionMetadata = Schema.Schema.Type<typeof SessionMetadata>

/**
 * CreateSession outcome. When `initial` is provided, the result discriminates
 * between full success, message-sent-but-promote-failed, and total failure.
 * This lets the client avoid restoring text when the message was actually sent.
 */
export const CreateSessionResult = Schema.Union(
  Schema.TaggedStruct("created", {
    metadata: SessionMetadata,
  }),
  Schema.TaggedStruct("created_message_failed", {
    sessionId: Schema.String,
    error: Schema.String,
  }),
  Schema.TaggedStruct("failed", {
    error: Schema.String,
  }),
)
export type CreateSessionResult = Schema.Schema.Type<typeof CreateSessionResult>

export const ListSessionsResult = Schema.Struct({
  items: Schema.Array(SessionMetadata),
  nextCursor: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  hasMore: Schema.Boolean
})
export type ListSessionsResult = Schema.Schema.Type<typeof ListSessionsResult>

export const SessionCwdSummary = Schema.Struct({
  cwd: Schema.String,
  updatedAt: Schema.Number,
  sessionCount: Schema.Number
})
export type SessionCwdSummary = Schema.Schema.Type<typeof SessionCwdSummary>

export const ActiveSessionStatus = Schema.Struct({
  sessionId: Schema.String,
  workStatus: Schema.Literal("idle", "working"),
  activeWorkerCount: Schema.Number,
  lastMessageAt: Schema.Number
})
export type ActiveSessionStatus = Schema.Schema.Type<typeof ActiveSessionStatus>

export const ActiveSessionStatuses = Schema.Struct({
  sessions: Schema.Array(ActiveSessionStatus)
})
export type ActiveSessionStatuses = Schema.Schema.Type<typeof ActiveSessionStatuses>
