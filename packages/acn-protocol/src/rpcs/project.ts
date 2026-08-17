import { Rpc } from "@effect/rpc"
import { Schema } from "effect"
import { ProjectError } from "../errors"
import {
  ListProjectsResultSchema,
  ProjectChangeSchema,
  ProjectIdSchema,
  ProjectRecordSchema,
} from "../schemas/project"
import { makeAcnSubscriptionRpc } from "./subscription"

export const ListProjects = Rpc.make("ListProjects", {
  payload: Schema.Struct({
    includeRemoved: Schema.optionalWith(Schema.Boolean, { default: () => false }),
  }),
  success: ListProjectsResultSchema,
  error: ProjectError,
})

export const CreateProject = Rpc.make("CreateProject", {
  payload: Schema.Struct({ sourceDirectory: Schema.String, name: Schema.String }),
  success: ProjectRecordSchema,
  error: ProjectError,
})

export const EditProject = Rpc.make("EditProject", {
  payload: Schema.Struct({
    projectId: ProjectIdSchema,
    name: Schema.String,
    sourceDirectory: Schema.String,
  }),
  success: ProjectRecordSchema,
  error: ProjectError,
})

export const RemoveProject = Rpc.make("RemoveProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectRecordSchema,
  error: ProjectError,
})

export const RestoreProject = Rpc.make("RestoreProject", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: ProjectRecordSchema,
  error: ProjectError,
})

export const RevealProjectSource = Rpc.make("RevealProjectSource", {
  payload: Schema.Struct({ projectId: ProjectIdSchema }),
  success: Schema.Struct({}),
  error: ProjectError,
})

export const StreamProjectChanges = makeAcnSubscriptionRpc("StreamProjectChanges", {
  payload: Schema.Struct({}),
  success: ProjectChangeSchema,
  error: ProjectError,
})
