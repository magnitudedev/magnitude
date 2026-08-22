import { Effect, Schema } from "effect"
import { Group, Mutation, Query, QueryClient, Subscription } from "@magnitudedev/effect-query"
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
import { ProjectIdSchema, type ProjectId } from "../schemas/project"
import { RelativePathSchema, parentDirectory, type RelativePath } from "../schemas/paths"
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

/**
 * Directory listings are kept fresh by `WatchProjectFiles` and by the
 * write/delete/move postconditions below; a short idle retention lets a
 * collapsed folder reopen without a round trip.
 */
const ListProjectDirectory = Query.make("ListProjectDirectory", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, directory: RelativePathSchema }),
  success: ProjectDirectoryListingSchema,
  error: ProjectFileAccessError,
  gcTime: "2 minutes",
})

/** Invalidation-only notifications for one project's source tree. */
const WatchProjectFiles = Subscription.make("WatchProjectFiles", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectFilesChangeSchema,
  error: ProjectFileAccessError,
  gcTime: "2 minutes",
})

const ReadProjectFile = Query.make("ReadProjectFile", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, path: RelativePathSchema }),
  success: ProjectFileSnapshotSchema,
  error: ProjectFileAccessError,
})

/** A committed write or deletion changes the exact file and its parent listing. */
const synchronizeProjectPath = (projectId: ProjectId, path: RelativePath) =>
  QueryClient.invalidate(ReadProjectFile.match({ projectId, path })).pipe(
    Effect.zipRight(QueryClient.invalidate(
      ListProjectDirectory.match({ projectId, directory: parentDirectory(path) }),
    )),
  )

const WriteProjectFile = Mutation.make("WriteProjectFile", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    content: Schema.String,
    expectedContentHash: FileContentHashSchema,
  }),
  success: ProjectFileTextSnapshotSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged, ProjectFileTooLarge),
  synchronize: (_, { projectId, path }) => synchronizeProjectPath(projectId, path),
})

const DeleteProjectFile = Mutation.make("DeleteProjectFile", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: RelativePathSchema,
    expectedContentHash: FileContentHashSchema,
  }),
  success: Schema.Struct({}),
  error: Schema.Union(ProjectFileAccessError, ProjectFileChanged),
  synchronize: (_, { projectId, path }) => synchronizeProjectPath(projectId, path),
})

/** A move changes source and destination listings; every listing and file of the project rereads. */
const MoveProjectEntry = Mutation.make("MoveProjectEntry", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    sourcePath: RelativePathSchema,
    destinationDirectory: RelativePathSchema,
  }),
  success: ProjectEntryMoveSchema,
  error: Schema.Union(ProjectFileAccessError, ProjectFileAlreadyExists),
  synchronize: () => QueryClient.invalidate(ListProjectDirectory.match()).pipe(
    Effect.zipRight(QueryClient.invalidate(ReadProjectFile.match())),
  ),
})

export const ProjectFiles = Group.make({
  ListProjectDirectory,
  WatchProjectFiles,
  ReadProjectFile,
  WriteProjectFile,
  DeleteProjectFile,
  MoveProjectEntry,
})
