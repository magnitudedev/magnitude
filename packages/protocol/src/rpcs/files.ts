import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import {
  ReadFilePayload,
  ReadFileResult,
  ResolvePathPayload,
  ResolvePathResult,
  SearchDirectoriesPayload,
  SearchDirectoriesResult,
  SearchMentionsPayload,
  SearchMentionsResult,
  WatchFilePayload,
  WatchFileWireEvent
} from "../schemas/files"
import { SessionError } from "../errors"

export const UploadAttachment = Rpc.make("UploadAttachment", {
  payload: Schema.Struct({
    sessionId: Schema.String,
    filename: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
    data: Schema.String,
    mediaType: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  }),
  success: Schema.Struct({
    path: Schema.String,
    filename: Schema.String,
  }),
  error: SessionError,
})

export const ListFiles = Rpc.make("ListFiles", {
  payload: Schema.Struct({
    cwd: Schema.String,
    glob: Schema.optional(Schema.String),
    limit: Schema.optionalWith(Schema.Number, { default: () => 100 })
  }),
  success: Schema.Array(Schema.String),
  error: SessionError
})

export const ReadFile = Rpc.make("ReadFile", {
  payload: ReadFilePayload,
  success: ReadFileResult,
  error: SessionError
})

export const CheckFileExists = Rpc.make("CheckFileExists", {
  payload: Schema.Struct({ cwd: Schema.String, path: Schema.String }),
  success: Schema.Boolean,
  error: SessionError
})

export const WatchFile = Rpc.make("WatchFile", {
  payload: WatchFilePayload,
  success: WatchFileWireEvent,
  error: SessionError,
  stream: true
})

export const ResolvePath = Rpc.make("ResolvePath", {
  payload: ResolvePathPayload,
  success: ResolvePathResult,
  error: SessionError
})

export const SearchMentions = Rpc.make("SearchMentions", {
  payload: SearchMentionsPayload,
  success: SearchMentionsResult,
  error: SessionError
})

export const SearchDirectories = Rpc.make("SearchDirectories", {
  payload: SearchDirectoriesPayload,
  success: SearchDirectoriesResult,
  error: SessionError
})
