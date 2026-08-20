import { Rpc } from "@effect/rpc"
import { GetGitRecentFilesPayload, GitRecentFilesSchema } from "../schemas/git"
import { InvalidDirectoryPath } from "../errors"

export const GetGitRecentFiles = Rpc.make("GetGitRecentFiles", {
  payload: GetGitRecentFilesPayload,
  success: GitRecentFilesSchema,
  error: InvalidDirectoryPath,
})
