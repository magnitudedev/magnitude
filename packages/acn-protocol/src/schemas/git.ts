import { Schema } from "effect"

export const GetGitRecentFilesPayload = Schema.Struct({
  cwd: Schema.String,
  limit: Schema.optionalWith(Schema.Number.pipe(Schema.int(), Schema.between(1, 100)), {
    default: () => 20,
  }),
})
export type GetGitRecentFilesPayload = Schema.Schema.Type<typeof GetGitRecentFilesPayload>

/**
 * Closed result union for recent-file discovery. Absence of Git, or a cwd
 * outside any repository, are normal states the caller renders (typically as
 * "no suggestions") — never silently caught into an empty list.
 */
export const GitRecentFilesSchema = Schema.Union(
  Schema.TaggedStruct("git_unavailable", {}),
  Schema.TaggedStruct("not_git_repository", {}),
  Schema.TaggedStruct("git_inspection_failed", {}),
  Schema.TaggedStruct("recent_git_files", { files: Schema.Array(Schema.String) }),
)
export type GitRecentFiles = typeof GitRecentFilesSchema.Type
