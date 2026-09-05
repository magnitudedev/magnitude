import { Rpc } from "@effect/rpc"
import { replaySafe, atMostOnce } from "../transport/recovery"
import { Schema } from "effect"
import {
  CreateSessionInitial,
  CreateSessionResult,
  ActiveSessionStatuses as ActiveSessionStatusesSchema,
  SessionPageSchema,
  SessionPageRequestSchema,
  RecentDirectoryPageRequestSchema,
  RecentDirectoryPageSchema,
  PreloadSessionResult,
  SessionMetadata,
  SessionOptions,
} from "../schemas/session"
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

const ListSessions = Rpc.make("ListSessions", {
  payload: SessionPageRequestSchema,
  success: SessionPageSchema,
  error: Schema.Union(InvalidSessionPageCursor, SessionInspectionUnavailable),
}).pipe(replaySafe)

const ListRecentSessionDirectories = Rpc.make("ListRecentSessionDirectories", {
  payload: RecentDirectoryPageRequestSchema,
  success: RecentDirectoryPageSchema,
  error: Schema.Union(InvalidDirectoryPageCursor, SessionInspectionUnavailable),
}).pipe(replaySafe)

const StreamActiveSessionStatuses = Rpc.make("StreamActiveSessionStatuses", {
  payload: Schema.Struct({}),
  success: ActiveSessionStatusesSchema,
  error: SessionError,
  stream: true,
})

const CreateSession = Rpc.make("CreateSession", {
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    initial: Schema.optionalWith(CreateSessionInitial, { as: "Option", exact: true }),
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: CreateSessionResult,
  error: SessionError,
}).pipe(atMostOnce)

const PreloadSession = Rpc.make("PreloadSession", {
  payload: Schema.Struct({
    cwd: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: PreloadSessionResult,
  error: SessionError,
}).pipe(atMostOnce)

const ReleaseSessionPreload = Rpc.make("ReleaseSessionPreload", {
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: SessionError,
}).pipe(replaySafe)

const GetSession = Rpc.make("GetSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: Schema.Union(SessionNotFound, SessionMetadataUnreadable),
}).pipe(replaySafe)

const DeleteArchivedSession = Rpc.make("DeleteArchivedSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: Schema.Struct({}),
  error: Schema.Union(
    SessionNotFound,
    SessionNotArchived,
    SessionMetadataUnreadable,
    SessionMetadataWriteFailed,
  ),
}).pipe(atMostOnce)

const SessionCommandError = Schema.Union(
  SessionNotFound,
  SessionMetadataUnreadable,
  SessionMetadataWriteFailed,
)

const ArchiveSession = Rpc.make("ArchiveSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
}).pipe(replaySafe)

const RestoreSession = Rpc.make("RestoreSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
}).pipe(replaySafe)

const SetSessionPinned = Rpc.make("SetSessionPinned", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    pinned: Schema.Boolean,
  }),
  success: SessionMetadata,
  error: SessionCommandError,
}).pipe(replaySafe)

export const Sessions = {
  listSessions: ListSessions,
  listRecentSessionDirectories: ListRecentSessionDirectories,
  streamActiveSessionStatuses: StreamActiveSessionStatuses,
  createSession: CreateSession,
  preloadSession: PreloadSession,
  releaseSessionPreload: ReleaseSessionPreload,
  getSession: GetSession,
  deleteArchivedSession: DeleteArchivedSession,
  archiveSession: ArchiveSession,
  restoreSession: RestoreSession,
  setSessionPinned: SetSessionPinned,
}
