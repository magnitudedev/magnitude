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
} from "../schemas/session"
import { makeAcnSubscriptionRpc } from "./subscription"
import { SessionError } from "../errors"

const ListSessionsPayloadFields = {
  cwd: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
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

export const DeleteSession = Rpc.make("DeleteSession", {
  payload: Schema.Struct({ sessionId: Schema.String }),
  success: Schema.Struct({}),
  error: SessionError
})
