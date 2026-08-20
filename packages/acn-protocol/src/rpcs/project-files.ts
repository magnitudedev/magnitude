import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { makeAcnSubscriptionRpc } from "./subscription"
import {
  DirectoryAccessDenied,
  DirectoryNotFound,
  FileSystemUnavailable,
  InvalidProjectFilePath,
  PathNotDirectory,
  ProjectFileAccessDenied,
  ProjectFileAlreadyExists,
  ProjectFileChanged,
  ProjectFileNotFound,
  ProjectFileTooLarge,
  ProjectNotFound,
  ProjectStoreUnavailable,
} from "../errors"
import { ProjectIdSchema } from "../schemas/project"
import { RelativePathSchema } from "../schemas/paths"
import {
  FileContentHashSchema,
  ProjectDirectoryListingSchema,
  ProjectEntryMoveSchema,
  ProjectFileSnapshotSchema,
  ProjectFileTextSnapshotSchema,
  ProjectFilesChangeSchema,
} from "../schemas/project-files"

/**
 * Failures shared by every project-file operation: resolving the Project,
 * opening its cwd, and resolving the contained path.
 */
const ProjectFileAccessError = Schema.Union(
  ProjectNotFound,
  ProjectStoreUnavailable,
  DirectoryNotFound,
  DirectoryAccessDenied,
  PathNotDirectory,
  FileSystemUnavailable,
  InvalidProjectFilePath,
  ProjectFileNotFound,
  ProjectFileAccessDenied,
)

export const ListProjectDirectory = Rpc.make("ListProjectDirectory", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, directory: RelativePathSchema }),
  success: ProjectDirectoryListingSchema,
  error: ProjectFileAccessError,
})

export const WatchProjectFiles = makeAcnSubscriptionRpc("WatchProjectFiles", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectFilesChangeSchema,
  error: ProjectFileAccessError,
})

export const ReadProjectFile = Rpc.make("ReadProjectFile", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, path: RelativePathSchema }),
  success: ProjectFileSnapshotSchema,
  error: ProjectFileAccessError,
})

export const WriteProjectFile = Rpc.make("WriteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    content: Schema.String,
    expectedContentHash: FileContentHashSchema,
  }),
  success: ProjectFileTextSnapshotSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged, ProjectFileTooLarge),
})

export const DeleteProjectFile = Rpc.make("DeleteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    expectedContentHash: FileContentHashSchema,
  }),
  success: Schema.Struct({}),
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged),
})

export const MoveProjectEntry = Rpc.make("MoveProjectEntry", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    sourcePath: RelativePathSchema,
    destinationDirectory: RelativePathSchema,
  }),
  success: ProjectEntryMoveSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileAlreadyExists),
})
