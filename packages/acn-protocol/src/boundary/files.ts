import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
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
} from "../errors"

/** Failures of a traversal rooted at the requested cwd. */
const TraversalError = Schema.Union(
  InvalidDirectoryPath,
  DirectoryNotFound,
  DirectoryAccessDenied,
  PathNotDirectory,
  FileSystemUnavailable,
)

const ListFiles = Rpc.make("ListFiles", {
  payload: Schema.Struct({
    cwd: Schema.String,
    glob: Schema.optional(Schema.String),
    limit: Schema.optionalWith(Schema.Number, { default: () => 100 }),
  }),
  success: Schema.Array(Schema.String),
  error: TraversalError,
}).pipe(replaySafe)

const ReadFile = Rpc.make("ReadFile", {
  payload: ReadFilePayload,
  success: ReadFileResult,
  error: Schema.Union(
    InvalidDirectoryPath,
    FileNotFound,
    FileAccessDenied,
    PathNotFile,
    FileSystemUnavailable,
  ),
}).pipe(replaySafe)

const CheckFileExists = Rpc.make("CheckFileExists", {
  payload: Schema.Struct({ cwd: Schema.String, path: Schema.String }),
  success: Schema.Boolean,
  error: InvalidDirectoryPath,
}).pipe(replaySafe)

const WatchFile = Rpc.make("WatchFile", {
  payload: WatchFilePayload,
  success: WatchFileEvent,
  error: InvalidDirectoryPath,
  stream: true,
})

const ResolvePath = Rpc.make("ResolvePath", {
  payload: ResolvePathPayload,
  success: ResolvePathResult,
  error: InvalidDirectoryPath,
}).pipe(replaySafe)

const SearchMentions = Rpc.make("SearchMentions", {
  payload: SearchMentionsPayload,
  success: SearchMentionsResult,
  error: TraversalError,
}).pipe(replaySafe)

const SearchDirectories = Rpc.make("SearchDirectories", {
  payload: SearchDirectoriesPayload,
  success: SearchDirectoriesResult,
  error: InvalidDirectoryPath,
}).pipe(replaySafe)

export const Files = {
  listFiles: ListFiles,
  readFile: ReadFile,
  checkFileExists: CheckFileExists,
  watchFile: WatchFile,
  resolvePath: ResolvePath,
  searchMentions: SearchMentions,
  searchDirectories: SearchDirectories,
}
