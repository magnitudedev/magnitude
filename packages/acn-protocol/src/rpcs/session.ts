import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import {
  CreateSessionInitial,
  CreateSessionResult,
  ActiveSessionStatuses,
  SessionPageSchema,
  SessionPageRequestSchema,
  RecentDirectoryPageRequestSchema,
  RecentDirectoryPageSchema,
  SessionChangeSchema,
  PreloadSessionResult,
  SessionMetadata,
  SessionOptions,
} from "../schemas/session"
import { makeAcnSubscriptionRpc } from "./subscription"
import {
  InvalidDirectoryPageCursor,
  InvalidSessionPageCursor,
  SessionError,
  SessionInspectionUnavailable,
  SessionMetadataUnreadable,
  SessionMetadataWriteFailed,
  SessionNotArchived,
  SessionNotFound,
} from "../errors"

export const ListSessions = Rpc.make("ListSessions", {
  payload: SessionPageRequestSchema,
  success: SessionPageSchema,
  error: Schema.Union(InvalidSessionPageCursor, SessionInspectionUnavailable),
})

export const ListRecentSessionDirectories = Rpc.make("ListRecentSessionDirectories", {
  payload: RecentDirectoryPageRequestSchema,
  success: RecentDirectoryPageSchema,
  error: Schema.Union(InvalidDirectoryPageCursor, SessionInspectionUnavailable),
})

export const StreamActiveSessionStatuses = makeAcnSubscriptionRpc("StreamActiveSessionStatuses", {
  payload: Schema.Struct({}),
  success: ActiveSessionStatuses,
  error: SessionError,
})

export const StreamSessionChanges = makeAcnSubscriptionRpc("StreamSessionChanges", {
  payload: Schema.Struct({}),
  success: SessionChangeSchema,
  error: SessionInspectionUnavailable,
})

export const CreateSession = Rpc.make("CreateSession", {
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    initial: Schema.optionalWith(CreateSessionInitial, { as: "Option", exact: true }),
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
  }),
  success: CreateSessionResult,
  error: SessionError
})

export const PreloadSession = Rpc.make("PreloadSession", {
  payload: Schema.Struct({
    cwd: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
  }),
  success: PreloadSessionResult,
  error: SessionError
})

export const ReleaseSessionPreload = Rpc.make("ReleaseSessionPreload", {
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
  }),
  success: Schema.Struct({}),
  error: SessionError
})

export const GetSession = Rpc.make("GetSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: Schema.Union(SessionNotFound, SessionMetadataUnreadable),
})

export const DeleteArchivedSession = Rpc.make("DeleteArchivedSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: Schema.Struct({}),
  error: Schema.Union(
    SessionNotFound,
    SessionNotArchived,
    SessionMetadataUnreadable,
    SessionMetadataWriteFailed,
  ),
})

const SessionCommandError = Schema.Union(
  SessionNotFound,
  SessionMetadataUnreadable,
  SessionMetadataWriteFailed,
)

export const ArchiveSession = Rpc.make("ArchiveSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
})

export const RestoreSession = Rpc.make("RestoreSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
})

export const SetSessionPinned = Rpc.make("SetSessionPinned", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    pinned: Schema.Boolean,
  }),
  success: SessionMetadata,
  error: SessionCommandError,
})
