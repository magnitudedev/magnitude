import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import {
  CreateSessionInitial,
  CreateSessionResult,
  ActiveSessionStatuses,
  ListSessionsResult,
  PreloadSessionResult,
  SessionCwdSummary,
  SessionMetadata,
  SessionOptions,
  SessionArchiveFilter,
} from "../schemas/session"
import { ProjectIdSchema } from "../schemas/project"
import { makeAcnSubscriptionRpc } from "./subscription"
import { SessionError } from "../errors"

const ListSessionsPayloadFields = {
  cwd: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  projectId: Schema.optionalWith(ProjectIdSchema, { as: "Option", exact: true }),
  archiveFilter: Schema.optionalWith(SessionArchiveFilter, { default: () => "active" }),
  prioritizePinned: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  query: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  cursor: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  limit: Schema.optionalWith(Schema.Number, { default: () => 50 })
}

export const ListSessions = Rpc.make("ListSessions", {
  payload: ListSessionsPayloadFields,
  success: ListSessionsResult,
  error: SessionError
})

export const ListSessionCwds = Rpc.make("ListSessionCwds", {
  payload: Schema.Struct({}),
  success: Schema.Array(SessionCwdSummary),
  error: SessionError
})

export const StreamActiveSessionStatuses = makeAcnSubscriptionRpc("StreamActiveSessionStatuses", {
  payload: Schema.Struct({}),
  success: ActiveSessionStatuses,
  error: SessionError,
})

export const CreateSession = Rpc.make("CreateSession", {
  payload: Schema.Struct({
    cwd: Schema.String,
    projectId: Schema.optionalWith(ProjectIdSchema, { as: "Option", exact: true }),
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
    projectId: Schema.optionalWith(ProjectIdSchema, { as: "Option", exact: true }),
    options: Schema.optionalWith(SessionOptions, { as: "Option", exact: true }),
    draftOwnerId: Schema.optionalWith(Schema.String, { as: "Option", exact: true })
  }),
  success: PreloadSessionResult,
  error: SessionError
})

export const ReleaseSessionPreload = Rpc.make("ReleaseSessionPreload", {
  payload: Schema.Struct({
    cwd: Schema.String,
    projectId: Schema.optionalWith(ProjectIdSchema, { as: "Option", exact: true }),
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
  error: SessionError
})

export const DeleteArchivedSession = Rpc.make("DeleteArchivedSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError
})

export const ArchiveSession = Rpc.make("ArchiveSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionError,
})

export const RestoreSession = Rpc.make("RestoreSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: SessionMetadata,
  error: SessionError,
})

export const SetSessionPinned = Rpc.make("SetSessionPinned", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    pinned: Schema.Boolean,
  }),
  success: SessionMetadata,
  error: SessionError,
})
