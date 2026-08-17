import { Schema } from "effect"
import { ProjectIdSchema } from "./schemas/project"
import { SlotIdSchema } from "./schemas/model-state"

export class SessionNotFound extends Schema.TaggedError<SessionNotFound>()(
  "SessionNotFound",
  { sessionId: Schema.String }
) {}

export class SessionAlreadyExists extends Schema.TaggedError<SessionAlreadyExists>()(
  "SessionAlreadyExists",
  { sessionId: Schema.String }
) {}

export class SessionStartFailed extends Schema.TaggedError<SessionStartFailed>()(
  "SessionStartFailed",
  { sessionId: Schema.String, reason: Schema.String }
) {}

export class SessionOperationFailed extends Schema.TaggedError<SessionOperationFailed>()(
  "SessionOperationFailed",
  { operation: Schema.String, reason: Schema.String }
) {}

export class DisplayViewNotOpen extends Schema.TaggedError<DisplayViewNotOpen>()(
  "DisplayViewNotOpen",
  { sessionId: Schema.String, viewId: Schema.String }
) {}

export class InvalidSessionPath extends Schema.TaggedError<InvalidSessionPath>()(
  "InvalidSessionPath",
  { path: Schema.String }
) {}

export const SessionError = Schema.Union(
  SessionNotFound,
  SessionAlreadyExists,
  SessionStartFailed,
  SessionOperationFailed,
  DisplayViewNotOpen,
  InvalidSessionPath
)
export type SessionError = Schema.Schema.Type<typeof SessionError>

export class ProjectNotFound extends Schema.TaggedError<ProjectNotFound>()(
  "ProjectNotFound",
  { projectId: ProjectIdSchema },
) {}

export class InvalidProjectName extends Schema.TaggedError<InvalidProjectName>()(
  "InvalidProjectName",
  { name: Schema.String },
) {}

export class InvalidProjectSource extends Schema.TaggedError<InvalidProjectSource>()(
  "InvalidProjectSource",
  { path: Schema.String, reason: Schema.String },
) {}

export class ProjectSourceAlreadyRegistered extends Schema.TaggedError<ProjectSourceAlreadyRegistered>()(
  "ProjectSourceAlreadyRegistered",
  { projectId: ProjectIdSchema, path: Schema.String },
) {}

export class ProjectBusy extends Schema.TaggedError<ProjectBusy>()(
  "ProjectBusy",
  { projectId: ProjectIdSchema },
) {}

export class ProjectOperationFailed extends Schema.TaggedError<ProjectOperationFailed>()(
  "ProjectOperationFailed",
  { operation: Schema.String, reason: Schema.String },
) {}

export const ProjectError = Schema.Union(
  ProjectNotFound,
  InvalidProjectName,
  InvalidProjectSource,
  ProjectSourceAlreadyRegistered,
  ProjectBusy,
  ProjectOperationFailed,
)
export type ProjectError = Schema.Schema.Type<typeof ProjectError>

export class LocalModelMutationFailed extends Schema.TaggedError<LocalModelMutationFailed>()(
  "LocalModelMutationFailed",
  {
    code: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  },
) {}

export class ModelSlotMutationRejected extends Schema.TaggedError<ModelSlotMutationRejected>()(
  "ModelSlotMutationRejected",
  {
    slotId: SlotIdSchema,
    message: Schema.String,
  },
) {}

export class ModelSlotMutationFailed extends Schema.TaggedError<ModelSlotMutationFailed>()(
  "ModelSlotMutationFailed",
  {
    slotId: SlotIdSchema,
    code: Schema.String,
    message: Schema.String,
    retryable: Schema.Boolean,
  },
) {}

export const ModelSlotUpdateError = Schema.Union(
  ModelSlotMutationRejected,
  ModelSlotMutationFailed,
)
export type ModelSlotUpdateError = Schema.Schema.Type<typeof ModelSlotUpdateError>

export class ModelPreferenceMutationFailed extends Schema.TaggedError<ModelPreferenceMutationFailed>()(
  "ModelPreferenceMutationFailed",
  { message: Schema.String },
) {}

export const LocalInferenceError = Schema.Union(
  LocalModelMutationFailed,
  ModelSlotMutationRejected,
  ModelSlotMutationFailed,
)
export type LocalInferenceError = Schema.Schema.Type<typeof LocalInferenceError>

export class OnboardingError extends Schema.TaggedError<OnboardingError>()(
  "OnboardingError",
  {
    operation: Schema.String,
    message: Schema.String,
  },
) {}
