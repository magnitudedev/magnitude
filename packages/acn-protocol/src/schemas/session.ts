import { Schema } from "effect"
import { RawMessageUploads, RawMentionOccurrence } from "./attachments"
import { DirectoryPathSchema } from "./paths"
export {
  RawClipboardImageAttachment,
  DisplayAttachment,
  RawFileImageAttachment,
  RawImageAttachment,
  RawMessageUpload,
  RawMessageUploads,
  RawTextFileUpload,
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
  uploads: RawMessageUploads,
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
  cwd: DirectoryPathSchema,
  archived: Schema.Boolean,
  pinnedAt: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  createdAt: Schema.Number,
  updatedAt: Schema.Number,
  messageCount: Schema.Number,
  lastMessage: Schema.Union(Schema.String, Schema.Null)
})
export type SessionMetadata = Schema.Schema.Type<typeof SessionMetadata>

export const SessionArchiveFilter = Schema.Literal("active", "archived", "all")
export type SessionArchiveFilter = Schema.Schema.Type<typeof SessionArchiveFilter>

export const SessionPinFilter = Schema.Literal("pinned", "unpinned", "all")
export type SessionPinFilter = Schema.Schema.Type<typeof SessionPinFilter>

export const SessionPageCursorSchema = Schema.String.pipe(Schema.brand("SessionPageCursor"))
export type SessionPageCursor = typeof SessionPageCursorSchema.Type

export const SessionPageRequestSchema = Schema.Struct({
  cwd: Schema.optionalWith(DirectoryPathSchema, { as: "Option", exact: true }),
  archive: Schema.optionalWith(SessionArchiveFilter, { default: () => "active" }),
  pin: Schema.optionalWith(SessionPinFilter, { default: () => "all" }),
  query: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  cursor: Schema.optionalWith(SessionPageCursorSchema, { as: "Option", exact: true }),
  limit: Schema.optionalWith(Schema.Number.pipe(Schema.int(), Schema.between(1, 100)), {
    default: () => 50,
  }),
})
export type SessionPageRequest = typeof SessionPageRequestSchema.Type

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

export const SessionPageSchema = Schema.Struct({
  items: Schema.Array(SessionMetadata),
  nextCursor: Schema.optionalWith(SessionPageCursorSchema, { as: "Option", exact: true }),
})
export type SessionPage = Schema.Schema.Type<typeof SessionPageSchema>

export const DirectoryPageCursorSchema = Schema.String.pipe(Schema.brand("DirectoryPageCursor"))
export type DirectoryPageCursor = typeof DirectoryPageCursorSchema.Type

export const RecentDirectoryPageRequestSchema = Schema.Struct({
  cursor: Schema.optionalWith(DirectoryPageCursorSchema, { as: "Option", exact: true }),
  limit: Schema.optionalWith(Schema.Number.pipe(Schema.int(), Schema.between(1, 100)), {
    default: () => 20,
  }),
})
export type RecentDirectoryPageRequest = typeof RecentDirectoryPageRequestSchema.Type

export const RecentDirectorySchema = Schema.Struct({
  cwd: DirectoryPathSchema,
  lastActiveAt: Schema.Number,
  sessionCount: Schema.Number.pipe(Schema.int(), Schema.positive()),
})
export type RecentDirectory = typeof RecentDirectorySchema.Type

export const RecentDirectoryPageSchema = Schema.Struct({
  items: Schema.Array(RecentDirectorySchema),
  nextCursor: Schema.optionalWith(DirectoryPageCursorSchema, { as: "Option", exact: true }),
})
export type RecentDirectoryPage = typeof RecentDirectoryPageSchema.Type

/** Invalidation-only session metadata change notification. */
export const SessionChangeSchema = Schema.Struct({})
export type SessionChange = typeof SessionChangeSchema.Type

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
