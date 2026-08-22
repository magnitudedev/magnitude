import { Schema } from "effect"
import { Group, Mutation, Query, Subscription } from "@magnitudedev/effect-query"
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
  WatchFileEvent,
} from "../schemas/files"
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

const UploadAttachment = Mutation.make("UploadAttachment", {
  policy: { recovery: "AtMostOnce" },
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

const ListFiles = Query.make("ListFiles", {
  payload: Schema.Struct({
    cwd: Schema.String,
    glob: Schema.optional(Schema.String),
    limit: Schema.optionalWith(Schema.Number, { default: () => 100 }),
  }),
  success: Schema.Array(Schema.String),
  error: TraversalError,
})

/** Kept fresh by `WatchFile` while the file is observed. */
const ReadFile = Query.make("ReadFile", {
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

const CheckFileExists = Query.make("CheckFileExists", {
  payload: Schema.Struct({ cwd: Schema.String, path: Schema.String }),
  success: Schema.Boolean,
  error: InvalidDirectoryPath,
})

/** Invalidation-only notifications for one host file. */
const WatchFile = Subscription.make("WatchFile", {
  payload: WatchFilePayload,
  success: WatchFileEvent,
  error: InvalidDirectoryPath,
  gcTime: "30 seconds",
})

const ResolvePath = Query.make("ResolvePath", {
  payload: ResolvePathPayload,
  success: ResolvePathResult,
  error: InvalidDirectoryPath,
})

/** Mention candidates for one query; results are not retained beyond the moment. */
const SearchMentions = Query.make("SearchMentions", {
  payload: SearchMentionsPayload,
  success: SearchMentionsResult,
  error: TraversalError,
  gcTime: "30 seconds",
})

const SearchDirectories = Query.make("SearchDirectories", {
  payload: SearchDirectoriesPayload,
  success: SearchDirectoriesResult,
  error: InvalidDirectoryPath,
  gcTime: "30 seconds",
})

export const Files = Group.make({
  UploadAttachment,
  ListFiles,
  ReadFile,
  CheckFileExists,
  WatchFile,
  ResolvePath,
  SearchMentions,
  SearchDirectories,
})
