import { Schema } from "effect"
import { ProjectIdSchema } from "./schemas/project"
import { RelativePathSchema } from "./schemas/paths"
import { ProjectFileTextSnapshotSchema } from "./schemas/project-files"
import { SlotIdSchema, UnsupportedReasonSchema } from "./schemas/model-state"
import { HuggingFaceAccessSchema } from "./schemas/hugging-face"

/** An operation whose handler failed: `operation` names it and `reason` says why. */
export class OperationFailed extends Schema.TaggedError<OperationFailed>()(
  "OperationFailed",
  { operation: Schema.String, reason: Schema.String }
) {}

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

/** The inference service could not say which models it serves. */
export class ServableModelsUnavailable extends Schema.TaggedError<ServableModelsUnavailable>()(
  "ServableModelsUnavailable",
  { message: Schema.String },
) {}

/** A Hugging Face link, repository, version, drafter reference, or local path that names nothing usable. */
export class InvalidModelReference extends Schema.TaggedError<InvalidModelReference>()(
  "InvalidModelReference",
  { reference: Schema.String, message: Schema.String },
) {}

/**
 * Why a model source couldn't be used: the inference service didn't answer, Hugging Face didn't
 * answer, the file couldn't be read, or the model isn't available.
 */
export const ModelSourceUnavailableReasonSchema = Schema.Literal(
  "ServiceUnreachable",
  "SourceUnreachable",
  "Unreadable",
  "Unavailable",
)
export type ModelSourceUnavailableReason = typeof ModelSourceUnavailableReasonSchema.Type

/** Hugging Face or a local file could not be read; `retryable` when the cause may clear. */
export class ModelSourceUnavailable extends Schema.TaggedError<ModelSourceUnavailable>()(
  "ModelSourceUnavailable",
  { reason: ModelSourceUnavailableReasonSchema, message: Schema.String, retryable: Schema.Boolean },
) {}

/**
 * A Hugging Face repository that Hugging Face won't show: it doesn't exist, or it's private or gated
 * and the request has no access. Hugging Face answers an anonymous request for a missing repository
 * as if it were private, so the two can't be told apart.
 */
export class ModelRepositoryNotFound extends Schema.TaggedError<ModelRepositoryNotFound>()(
  "ModelRepositoryNotFound",
  { reference: Schema.String, message: Schema.String },
) {}

/**
 * A gated Hugging Face repository refused its files: Magnitude isn't signed in to Hugging Face
 * (`SignIn`), or the signed-in account hasn't accepted the repository's terms or been approved
 * (`AcceptTerms`).
 */
export class HuggingFaceAccessRequired extends Schema.TaggedError<HuggingFaceAccessRequired>()(
  "HuggingFaceAccessRequired",
  { repository: Schema.String, access: HuggingFaceAccessSchema, message: Schema.String },
) {}

/** A Hugging Face sign-in couldn't start or a sign-out couldn't be saved. */
export class HuggingFaceSignInFailed extends Schema.TaggedError<HuggingFaceSignInFailed>()(
  "HuggingFaceSignInFailed",
  { message: Schema.String },
) {}

export const ModelCandidateError = Schema.Union(
  InvalidModelReference,
  ModelRepositoryNotFound,
  ModelSourceUnavailable,
  HuggingFaceAccessRequired,
)
export type ModelCandidateError = Schema.Schema.Type<typeof ModelCandidateError>

/**
 * A version refused before anything downloads, or a model refused a load: the engine can't run it, or it
 * needs more memory than this computer can provide and the request didn't force it. Clients word it from
 * these facts.
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

/**
 * A drafter the engine won't bind to this model, with the engine's reasons; no reasons means it
 * wasn't checked. Clients word it from these facts.
 */
export class DrafterNotUsable extends Schema.TaggedError<DrafterNotUsable>()(
  "DrafterNotUsable",
  { reference: Schema.String, reasons: Schema.Array(UnsupportedReasonSchema) },
) {}

export const CustomModelMutationError = Schema.Union(
  InvalidModelReference,
  ModelRepositoryNotFound,
  ModelSourceUnavailable,
  HuggingFaceAccessRequired,
  LocalModelMutationFailed,
  ModelVersionNotRunnable,
  DrafterNotUsable,
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
