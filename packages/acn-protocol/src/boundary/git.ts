import { Rpc } from "@effect/rpc"
import { replaySafe } from "../transport/recovery"
import { GetGitRecentFilesPayload, GitRecentFilesSchema } from "../schemas/git"
import { InvalidDirectoryPath } from "../errors"

const GetGitRecentFiles = Rpc.make("GetGitRecentFiles", {
  payload: GetGitRecentFilesPayload,
  success: GitRecentFilesSchema,
  error: InvalidDirectoryPath,
}).pipe(replaySafe)

export const Git = {
  getGitRecentFiles: GetGitRecentFiles,
}
