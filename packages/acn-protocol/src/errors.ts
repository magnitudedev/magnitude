import { Schema } from "effect"
import { ProjectIdSchema } from "./schemas/project"
import { RelativePathSchema } from "./schemas/paths"
import { ProjectFileTextSnapshotSchema } from "./schemas/project-files"
import { SlotIdSchema, UnsupportedReasonSchema } from "./schemas/model-state"

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

export class SessionNotArchived extends Schema.TaggedError<SessionNotArchived>()(
  "SessionNotArchived",
  { sessionId: Schema.String }
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
  SessionNotArchived,
  DisplayViewNotOpen,
  InvalidSessionPath
)
export type SessionError = Schema.Schema.Type<typeof SessionError>

export class ProjectNotFound extends Schema.TaggedError<ProjectNotFound>()(
  "ProjectNotFound",
  { projectId: ProjectIdSchema },
) {}

export class ProjectStoreUnavailable extends Schema.TaggedError<ProjectStoreUnavailable>()(
  "ProjectStoreUnavailable",
  {},
) {}

export class InvalidProjectPageCursor extends Schema.TaggedError<InvalidProjectPageCursor>()(
  "InvalidProjectPageCursor",
  {},
) {}

export class InvalidSessionPageCursor extends Schema.TaggedError<InvalidSessionPageCursor>()(
  "InvalidSessionPageCursor",
  {},
) {}

export class InvalidDirectoryPageCursor extends Schema.TaggedError<InvalidDirectoryPageCursor>()(
  "InvalidDirectoryPageCursor",
  {},
) {}

export class SessionMetadataWriteFailed extends Schema.TaggedError<SessionMetadataWriteFailed>()(
  "SessionMetadataWriteFailed",
  { sessionId: Schema.String },
) {}

export class SessionMetadataUnreadable extends Schema.TaggedError<SessionMetadataUnreadable>()(
  "SessionMetadataUnreadable",
  { sessionId: Schema.String },
) {}

export class SessionInspectionUnavailable extends Schema.TaggedError<SessionInspectionUnavailable>()(
  "SessionInspectionUnavailable",
  {},
) {}

export class InvalidProjectName extends Schema.TaggedError<InvalidProjectName>()(
  "InvalidProjectName",
  { name: Schema.String },
) {}

export class ProjectCwdAlreadyRegistered extends Schema.TaggedError<ProjectCwdAlreadyRegistered>()(
  "ProjectCwdAlreadyRegistered",
  { projectId: ProjectIdSchema, cwd: Schema.String },
) {}

export class RevealUnsupported extends Schema.TaggedError<RevealUnsupported>()(
  "RevealUnsupported",
  {},
) {}

export class RevealFailed extends Schema.TaggedError<RevealFailed>()(
  "RevealFailed",
  { path: Schema.String },
) {}

export class InvalidDirectoryPath extends Schema.TaggedError<InvalidDirectoryPath>()(
  "InvalidDirectoryPath",
  { path: Schema.String },
) {
  override get message(): string {
    return `${this.path} isn't an absolute folder path.`
  }
}

export class DirectoryNotFound extends Schema.TaggedError<DirectoryNotFound>()(
  "DirectoryNotFound",
  { path: Schema.String },
) {
  override get message(): string {
    return `${this.path} doesn't exist.`
  }
}

export class DirectoryAccessDenied extends Schema.TaggedError<DirectoryAccessDenied>()(
  "DirectoryAccessDenied",
  { path: Schema.String },
) {
  override get message(): string {
    return `Magnitude doesn't have permission to read ${this.path}.`
  }
}

export class PathNotDirectory extends Schema.TaggedError<PathNotDirectory>()(
  "PathNotDirectory",
  { path: Schema.String },
) {
  override get message(): string {
    return `${this.path} isn't a folder.`
  }
}

export class FileSystemUnavailable extends Schema.TaggedError<FileSystemUnavailable>()(
  "FileSystemUnavailable",
  { path: Schema.String },
) {}

export class InvalidRelativePath extends Schema.TaggedError<InvalidRelativePath>()(
  "InvalidRelativePath",
  { path: Schema.String },
) {}

export class FileNotFound extends Schema.TaggedError<FileNotFound>()(
  "FileNotFound",
  { path: Schema.String },
) {}

export class FileAccessDenied extends Schema.TaggedError<FileAccessDenied>()(
  "FileAccessDenied",
  { path: Schema.String },
) {}

export class PathNotFile extends Schema.TaggedError<PathNotFile>()(
  "PathNotFile",
  { path: Schema.String },
) {}

export class FileAlreadyExists extends Schema.TaggedError<FileAlreadyExists>()(
  "FileAlreadyExists",
  { path: Schema.String },
) {}

export class InvalidProjectFilePath extends Schema.TaggedError<InvalidProjectFilePath>()(
  "InvalidProjectFilePath",
  {
    path: RelativePathSchema,
    kind: Schema.Literal("empty_path", "escapes_root", "root_immovable"),
  },
) {}

export class ProjectFileNotFound extends Schema.TaggedError<ProjectFileNotFound>()(
  "ProjectFileNotFound",
  { path: RelativePathSchema },
) {}

export class ProjectFileAlreadyExists extends Schema.TaggedError<ProjectFileAlreadyExists>()(
  "ProjectFileAlreadyExists",
  { path: RelativePathSchema },
) {}

export class ProjectFileAccessDenied extends Schema.TaggedError<ProjectFileAccessDenied>()(
  "ProjectFileAccessDenied",
  {
    path: RelativePathSchema,
    kind: Schema.Literal(
      "symlink",
      "permission_denied",
      "not_regular_file",
      "not_directory",
      "not_text",
      "changed_on_disk",
      "already_in_destination",
      "self_move",
    ),
  },
) {}

export class ProjectFileTooLarge extends Schema.TaggedError<ProjectFileTooLarge>()(
  "ProjectFileTooLarge",
  { path: RelativePathSchema, size: Schema.Number },
) {}

export class ProjectFileChanged extends Schema.TaggedError<ProjectFileChanged>()(
  "ProjectFileChanged",
  { path: RelativePathSchema, current: ProjectFileTextSnapshotSchema },
) {}

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

/** A Hugging Face link, repository, version, drafter reference, or local path that names nothing usable. */
export class InvalidModelReference extends Schema.TaggedError<InvalidModelReference>()(
  "InvalidModelReference",
  { reference: Schema.String, message: Schema.String },
) {}

/**
 * Why a model source couldn't be used: the inference service didn't answer, Hugging Face didn't
 * answer, the repository needs accepted terms and an access token, the file couldn't be read, or
 * the model isn't available.
 */
export const ModelSourceUnavailableReasonSchema = Schema.Literal(
  "ServiceUnreachable",
  "SourceUnreachable",
  "AccessRequired",
  "Unreadable",
  "Unavailable",
)
export type ModelSourceUnavailableReason = typeof ModelSourceUnavailableReasonSchema.Type

/** Hugging Face or a local file could not be read; `retryable` when the cause may clear. */
export class ModelSourceUnavailable extends Schema.TaggedError<ModelSourceUnavailable>()(
  "ModelSourceUnavailable",
  { reason: ModelSourceUnavailableReasonSchema, message: Schema.String, retryable: Schema.Boolean },
) {}

export const ModelCandidateError = Schema.Union(
  InvalidModelReference,
  ModelSourceUnavailable,
)
export type ModelCandidateError = Schema.Schema.Type<typeof ModelCandidateError>

/**
 * A version refused before anything downloads: the engine can't run it, or it needs more memory than this
 * computer can provide and the request didn't force it. Clients word it from these facts.
 */
export class ModelVersionNotRunnable extends Schema.TaggedError<ModelVersionNotRunnable>()(
  "ModelVersionNotRunnable",
  {
    reference: Schema.String,
    version: Schema.String,
    verdict: Schema.Union(
      Schema.TaggedStruct("Unsupported", { reasons: Schema.NonEmptyArray(UnsupportedReasonSchema) }),
      Schema.TaggedStruct("DoesNotFit", { deficitBytes: Schema.NonNegativeInt }),
    ),
  },
) {}

export const CustomModelMutationError = Schema.Union(
  InvalidModelReference,
  ModelSourceUnavailable,
  LocalModelMutationFailed,
  ModelVersionNotRunnable,
)
export type CustomModelMutationError = Schema.Schema.Type<typeof CustomModelMutationError>

export const ModelLocationError = Schema.Union(
  InvalidDirectoryPath,
  DirectoryNotFound,
  DirectoryAccessDenied,
  PathNotDirectory,
  LocalModelMutationFailed,
)
export type ModelLocationError = Schema.Schema.Type<typeof ModelLocationError>
