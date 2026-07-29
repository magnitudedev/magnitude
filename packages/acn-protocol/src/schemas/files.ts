import { Schema } from "effect"

export const ReadFileFormat = Schema.Literal("text", "base64")
export type ReadFileFormat = Schema.Schema.Type<typeof ReadFileFormat>

export const ListFilesPayload = Schema.Struct({
  cwd: Schema.String,
  glob: Schema.optional(Schema.String),
  limit: Schema.optionalWith(Schema.Number, { default: () => 100 })
})
export type ListFilesPayload = Schema.Schema.Type<typeof ListFilesPayload>

export const ReadFilePayload = Schema.Struct({
  cwd: Schema.String,
  path: Schema.String,
  format: Schema.optionalWith(ReadFileFormat, { default: () => "text" }),
  offset: Schema.optionalWith(Schema.Number, { default: () => 0 })
})
export type ReadFilePayload = Schema.Schema.Type<typeof ReadFilePayload>

export const ReadFileResult = Schema.Struct({
  path: Schema.String,
  content: Schema.String,
  format: ReadFileFormat
})
export type ReadFileResult = Schema.Schema.Type<typeof ReadFileResult>

export const CheckFileExistsPayload = Schema.Struct({
  cwd: Schema.String,
  path: Schema.String
})
export type CheckFileExistsPayload = Schema.Schema.Type<typeof CheckFileExistsPayload>

export const WatchFileEvent = Schema.Struct({
  event: Schema.Literal("created", "changed", "removed"),
  path: Schema.String
})
export type WatchFileEvent = Schema.Schema.Type<typeof WatchFileEvent>

export const WatchFilePayload = Schema.Struct({
  cwd: Schema.String,
  path: Schema.String
})
export type WatchFilePayload = Schema.Schema.Type<typeof WatchFilePayload>

export const ResolvePathPayload = Schema.Struct({
  cwd: Schema.String,
  path: Schema.String,
  checkExists: Schema.optionalWith(Schema.Boolean, { default: () => true })
})
export type ResolvePathPayload = Schema.Schema.Type<typeof ResolvePathPayload>

export const ResolvePathResult = Schema.Struct({
  resolved: Schema.String,
  exists: Schema.Boolean,
  isDirectory: Schema.Boolean
})
export type ResolvePathResult = Schema.Schema.Type<typeof ResolvePathResult>

export const MentionLineRange = Schema.Struct({
  start: Schema.Number,
  end: Schema.Number
})
export type MentionLineRange = Schema.Schema.Type<typeof MentionLineRange>

export const MentionContentType = Schema.Literal("text", "directory")
export type MentionContentType = Schema.Schema.Type<typeof MentionContentType>

export const MentionCandidate = Schema.Struct({
  path: Schema.String,
  kind: Schema.Literal("file", "directory"),
  contentType: MentionContentType,
  warning: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  lineRange: Schema.optional(MentionLineRange)
})
export type MentionCandidate = Schema.Schema.Type<typeof MentionCandidate>

export const SearchMentionsPayload = Schema.Struct({
  cwd: Schema.String,
  query: Schema.String,
  limit: Schema.optionalWith(Schema.Number, { default: () => 40 }),
  visibleLimit: Schema.optionalWith(Schema.Number, { default: () => 10 }),
  includeRecent: Schema.optionalWith(Schema.Boolean, { default: () => true })
})
export type SearchMentionsPayload = Schema.Schema.Type<typeof SearchMentionsPayload>

export const SearchMentionsResult = Schema.Struct({
  query: Schema.String,
  lineRange: Schema.optional(MentionLineRange),
  candidates: Schema.Array(MentionCandidate),
  recentCandidates: Schema.Array(MentionCandidate),
  overflowCount: Schema.Number
})
export type SearchMentionsResult = Schema.Schema.Type<typeof SearchMentionsResult>

export const DirectoryCandidateSource = Schema.Literal("recent", "filesystem", "exact")
export type DirectoryCandidateSource = Schema.Schema.Type<typeof DirectoryCandidateSource>

export const DirectoryCandidate = Schema.Struct({
  path: Schema.String,
  label: Schema.String,
  source: DirectoryCandidateSource,
  lastActivity: Schema.optional(Schema.Number)
})
export type DirectoryCandidate = Schema.Schema.Type<typeof DirectoryCandidate>

export const SearchDirectoriesPayload = Schema.Struct({
  query: Schema.String,
  limit: Schema.optionalWith(Schema.Number, { default: () => 20 }),
  includeRecent: Schema.optionalWith(Schema.Boolean, { default: () => true })
})
export type SearchDirectoriesPayload = Schema.Schema.Type<typeof SearchDirectoriesPayload>

export const SearchDirectoriesResult = Schema.Struct({
  query: Schema.String,
  candidates: Schema.Array(DirectoryCandidate)
})
export type SearchDirectoriesResult = Schema.Schema.Type<typeof SearchDirectoriesResult>
