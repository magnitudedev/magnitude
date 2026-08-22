import { Group, Query } from "@magnitudedev/effect-query"
import { GetGitRecentFilesPayload, GitRecentFilesSchema } from "../schemas/git"
import { InvalidDirectoryPath } from "../errors"

const GetGitRecentFiles = Query.make("GetGitRecentFiles", {
  payload: GetGitRecentFilesPayload,
  success: GitRecentFilesSchema,
  error: InvalidDirectoryPath,
})

export const Git = Group.make({ GetGitRecentFiles })
