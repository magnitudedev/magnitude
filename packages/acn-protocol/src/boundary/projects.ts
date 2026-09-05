import { Rpc } from "@effect/rpc"
import { replaySafe, atMostOnce } from "../transport/recovery"
import { Schema } from "effect"
import {
  InvalidProjectName,
  InvalidDirectoryPath,
  DirectoryNotFound,
  DirectoryAccessDenied,
  PathNotDirectory,
  FileSystemUnavailable,
  InvalidProjectPageCursor,
  ProjectCwdAlreadyRegistered,
  ProjectNotFound,
  ProjectStoreUnavailable,
  RevealFailed,
  RevealUnsupported,
} from "../errors"
import {
  ProjectInspectionSchema,
  ProjectPageRequestSchema,
  ProjectPageSchema,
  ProjectIdSchema,
  ProjectSchema,
} from "../schemas/project"

const ListProjects = Rpc.make("ListProjects", {
  payload: ProjectPageRequestSchema,
  success: ProjectPageSchema,
  error: Schema.Union(InvalidProjectPageCursor, ProjectStoreUnavailable),
}).pipe(replaySafe)

const CreateProject = Rpc.make("CreateProject", {
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: ProjectSchema,
  error: Schema.Union(InvalidProjectName, InvalidDirectoryPath, DirectoryNotFound, DirectoryAccessDenied, PathNotDirectory, FileSystemUnavailable, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
}).pipe(replaySafe)

const EditProject = Rpc.make("EditProject", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    name: Schema.String,
    cwd: Schema.String,
  }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, InvalidProjectName, InvalidDirectoryPath, DirectoryNotFound, DirectoryAccessDenied, PathNotDirectory, FileSystemUnavailable, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
}).pipe(replaySafe)

const RemoveProject = Rpc.make("RemoveProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable),
}).pipe(replaySafe)

const RestoreProject = Rpc.make("RestoreProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
}).pipe(replaySafe)

const RevealProjectSource = Rpc.make("RevealProjectSource", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: Schema.Struct({}),
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable, RevealUnsupported, RevealFailed),
}).pipe(atMostOnce)

const InspectProject = Rpc.make("InspectProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectInspectionSchema,
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable),
}).pipe(replaySafe)

export const Projects = {
  listProjects: ListProjects,
  createProject: CreateProject,
  editProject: EditProject,
  removeProject: RemoveProject,
  restoreProject: RestoreProject,
  revealProjectSource: RevealProjectSource,
  inspectProject: InspectProject,
}
