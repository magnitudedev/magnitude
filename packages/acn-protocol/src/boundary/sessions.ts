import { Schema } from "effect"
import { Group, Mutation, Query } from "@magnitudedev/effect-query"
import {
  CreateSessionInitial,
  CreateSessionResult,
  ActiveSessionStatuses as ActiveSessionStatusesSchema,
  type ActiveSessionStatuses as ActiveSessionStatusesSnapshot,
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

/** Session pages are fresh until the ACN publishes a session change on `StreamChanges`. */
const ListSessions = Query.make("ListSessions", {
  payload: SessionPageRequestSchema,
  success: SessionPageSchema,
  error: Schema.Union(InvalidSessionPageCursor, SessionInspectionUnavailable),
  staleTime: Infinity,
})

const ListRecentSessionDirectories = Query.make("ListRecentSessionDirectories", {
  payload: RecentDirectoryPageRequestSchema,
  success: RecentDirectoryPageSchema,
  error: Schema.Union(InvalidDirectoryPageCursor, SessionInspectionUnavailable),
  staleTime: Infinity,
})

/** The resident-session status snapshot, folded from the ACN's status stream. */
const StreamActiveSessionStatuses = Query.fromStream("StreamActiveSessionStatuses", {
  payload: Schema.Struct({}),
  success: ActiveSessionStatusesSchema,
  error: SessionError,
  reduce: (_, snapshot): ActiveSessionStatusesSnapshot => snapshot,
})

const CreateSession = Mutation.make("CreateSession", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    initial: Schema.optionalWith(CreateSessionInitial, { as: "Option", exact: true }),
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: CreateSessionResult,
  error: SessionError,
})

const PreloadSession = Mutation.make("PreloadSession", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    cwd: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: PreloadSessionResult,
  error: SessionError,
})

const ReleaseSessionPreload = Mutation.make("ReleaseSessionPreload", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({
    cwd: Schema.String,
    sessionId: Schema.String,
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({}),
  error: SessionError,
})

const GetSession = Query.make("GetSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: Schema.Union(SessionNotFound, SessionMetadataUnreadable),
  staleTime: Infinity,
})

const DeleteArchivedSession = Mutation.make("DeleteArchivedSession", {
  policy: { recovery: "AtMostOnce" },
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

const ArchiveSession = Mutation.make("ArchiveSession", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
})

const RestoreSession = Mutation.make("RestoreSession", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionCommandError,
})

const SetSessionPinned = Mutation.make("SetSessionPinned", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({
    sessionId: Schema.String,
    pinned: Schema.Boolean,
  }),
  success: SessionMetadata,
  error: SessionCommandError,
})

export const Sessions = Group.make({
  ListSessions,
  ListRecentSessionDirectories,
  StreamActiveSessionStatuses,
  CreateSession,
  PreloadSession,
  ReleaseSessionPreload,
  GetSession,
  DeleteArchivedSession,
  ArchiveSession,
  RestoreSession,
  SetSessionPinned,
})
