import { Schema } from "effect"
import { Group, Mutation, Query } from "@magnitudedev/effect-query"
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

/** Project pages are fresh until the ACN publishes a project change on `StreamChanges`. */
const ListProjects = Query.make("ListProjects", {
  payload: ProjectPageRequestSchema,
  success: ProjectPageSchema,
  error: Schema.Union(InvalidProjectPageCursor, ProjectStoreUnavailable),
  staleTime: Infinity,
})

const CreateProject = Mutation.make("CreateProject", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ cwd: Schema.String, name: Schema.String }),
  success: ProjectSchema,
  error: Schema.Union(InvalidProjectName, InvalidDirectoryPath, DirectoryNotFound, DirectoryAccessDenied, PathNotDirectory, FileSystemUnavailable, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
})

const EditProject = Mutation.make("EditProject", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    name: Schema.String,
    cwd: Schema.String,
  }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, InvalidProjectName, InvalidDirectoryPath, DirectoryNotFound, DirectoryAccessDenied, PathNotDirectory, FileSystemUnavailable, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
})

const RemoveProject = Mutation.make("RemoveProject", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable),
})

const RestoreProject = Mutation.make("RestoreProject", {
  policy: { recovery: "ReplaySafe" },
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectSchema,
  error: Schema.Union(ProjectNotFound, ProjectCwdAlreadyRegistered, ProjectStoreUnavailable),
})

const RevealProjectSource = Mutation.make("RevealProjectSource", {
  policy: { recovery: "AtMostOnce" },
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: Schema.Struct({}),
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable, RevealUnsupported, RevealFailed),
})

const InspectProject = Query.make("InspectProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectInspectionSchema,
  error: Schema.Union(ProjectNotFound, ProjectStoreUnavailable),
  staleTime: Infinity,
})

export const Projects = Group.make({
  ListProjects,
  CreateProject,
  EditProject,
  RemoveProject,
  RestoreProject,
  RevealProjectSource,
  InspectProject,
})
