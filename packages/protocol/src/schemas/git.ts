import { Schema } from "effect"

export const GetGitRecentFilesPayload = Schema.Struct({
  cwd: Schema.String,
  limit: Schema.optionalWith(Schema.Number, { default: () => 20 })
})
export type GetGitRecentFilesPayload = Schema.Schema.Type<typeof GetGitRecentFilesPayload>
