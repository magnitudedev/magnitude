import { Rpc } from "@effect/rpc"
import { replaySafe, atMostOnce } from "../transport/recovery"
import { Schema } from "effect"
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

const ListProjectDirectory = Rpc.make("ListProjectDirectory", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, directory: RelativePathSchema }),
  success: ProjectDirectoryListingSchema,
  error: ProjectFileAccessError,
}).pipe(replaySafe)

const WatchProjectFiles = Rpc.make("WatchProjectFiles", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectFilesChangeSchema,
  error: ProjectFileAccessError,
  stream: true,
})

const ReadProjectFile = Rpc.make("ReadProjectFile", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, path: RelativePathSchema }),
  success: ProjectFileSnapshotSchema,
  error: ProjectFileAccessError,
}).pipe(replaySafe)

const WriteProjectFile = Rpc.make("WriteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    content: Schema.String,
    expectedContentHash: FileContentHashSchema,
  }),
  success: ProjectFileTextSnapshotSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged, ProjectFileTooLarge),
}).pipe(atMostOnce)

const DeleteProjectFile = Rpc.make("DeleteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    expectedContentHash: FileContentHashSchema,
  }),
  success: Schema.Struct({}),
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged),
}).pipe(atMostOnce)

const MoveProjectEntry = Rpc.make("MoveProjectEntry", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    sourcePath: RelativePathSchema,
    destinationDirectory: RelativePathSchema,
  }),
  success: ProjectEntryMoveSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileAlreadyExists),
}).pipe(atMostOnce)

export const ProjectFiles = {
  listProjectDirectory: ListProjectDirectory,
  watchProjectFiles: WatchProjectFiles,
  readProjectFile: ReadProjectFile,
  writeProjectFile: WriteProjectFile,
  deleteProjectFile: DeleteProjectFile,
  moveProjectEntry: MoveProjectEntry,
}
