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
  WatchFileEvent
} from "../schemas/files"
import { makeAcnSubscriptionRpc } from "./subscription"
import {
  DirectoryAccessDenied,
  DirectoryNotFound,
  FileAccessDenied,
  FileNotFound,
  FileSystemUnavailable,
  InvalidDirectoryPath,
  PathNotDirectory,
  PathNotFile,
  SessionError,
} from "../errors"

/** Failures of a traversal rooted at the requested cwd. */
const TraversalError = Schema.Union(
  InvalidDirectoryPath,
  DirectoryNotFound,
  DirectoryAccessDenied,
  PathNotDirectory,
  FileSystemUnavailable,
)

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
  error: TraversalError,
})

export const ReadFile = Rpc.make("ReadFile", {
  payload: ReadFilePayload,
  success: ReadFileResult,
  error: Schema.Union(
    InvalidDirectoryPath,
    FileNotFound,
    FileAccessDenied,
    PathNotFile,
    FileSystemUnavailable,
  ),
})

export const CheckFileExists = Rpc.make("CheckFileExists", {
  payload: Schema.Struct({ cwd: Schema.String, path: Schema.String }),
  success: Schema.Boolean,
  error: InvalidDirectoryPath,
})

export const WatchFile = makeAcnSubscriptionRpc("WatchFile", {
  payload: WatchFilePayload,
  success: WatchFileEvent,
  error: InvalidDirectoryPath,
})

export const ResolvePath = Rpc.make("ResolvePath", {
  payload: ResolvePathPayload,
  success: ResolvePathResult,
  error: InvalidDirectoryPath,
})

export const SearchMentions = Rpc.make("SearchMentions", {
  payload: SearchMentionsPayload,
  success: SearchMentionsResult,
  error: TraversalError,
})

export const SearchDirectories = Rpc.make("SearchDirectories", {
  payload: SearchDirectoriesPayload,
  success: SearchDirectoriesResult,
  error: InvalidDirectoryPath,
})
