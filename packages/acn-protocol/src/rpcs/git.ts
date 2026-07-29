import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { GetGitRecentFilesPayload } from "../schemas/git"
import { SessionError } from "../errors"

export const GetGitRecentFilesPayloadSchema = Schema.Struct({
  cwd: Schema.String,
  limit: Schema.optionalWith(Schema.Number, { default: () => 20 })
})

export const GetGitRecentFiles = Rpc.make("GetGitRecentFiles", {
  payload: GetGitRecentFilesPayloadSchema,
  success: Schema.Array(Schema.String),
  error: SessionError
})
