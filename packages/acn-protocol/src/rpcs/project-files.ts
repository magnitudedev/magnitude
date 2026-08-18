import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProjectFileError } from "../errors"
import { ProjectIdSchema } from "../schemas/project"
import {
  ProjectDirectoryListingSchema,
  ProjectEntryMoveSchema,
  ProjectFileRevisionSchema,
  ProjectFileSnapshotSchema,
  ProjectFileTextSnapshotSchema,
  ProjectFilesChangeSchema,
  ProjectRelativePathSchema,
} from "../schemas/project-files"
import { makeAcnSubscriptionRpc } from "./subscription"

const projectPathPayload = Schema.Struct({ projectId: ProjectIdSchema, path: ProjectRelativePathSchema })

export const ListProjectDirectory = Rpc.make("ListProjectDirectory", {
  payload: Schema.Struct({ projectId: ProjectIdSchema, directory: ProjectRelativePathSchema }),
  success: ProjectDirectoryListingSchema,
  error: ProjectFileError,
})

export const WatchProjectFiles = makeAcnSubscriptionRpc("WatchProjectFiles", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectFilesChangeSchema,
  error: ProjectFileError,
})

export const ReadProjectFile = Rpc.make("ReadProjectFile", {
  payload: projectPathPayload,
  success: ProjectFileSnapshotSchema,
  error: ProjectFileError,
})

export const WriteProjectFile = Rpc.make("WriteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: ProjectRelativePathSchema,
    content: Schema.String,
    expectedRevision: ProjectFileRevisionSchema,
  }),
  success: ProjectFileTextSnapshotSchema,
  error: ProjectFileError,
})

export const DeleteProjectFile = Rpc.make("DeleteProjectFile", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    path: ProjectRelativePathSchema,
    expectedRevision: ProjectFileRevisionSchema,
  }),
  success: Schema.Struct({}),
  error: ProjectFileError,
})

export const MoveProjectEntry = Rpc.make("MoveProjectEntry", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    sourcePath: ProjectRelativePathSchema,
    destinationDirectory: ProjectRelativePathSchema,
  }),
  success: ProjectEntryMoveSchema,
  error: ProjectFileError,
})
