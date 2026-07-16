import { Schema } from "effect"
import { HuggingFaceFilePath, HuggingFaceRepositoryId, HuggingFaceRevision } from "./identity"

export const HuggingFaceHubOperation = Schema.Literal("search", "resolve-revision", "resolve-paths", "download")
export type HuggingFaceHubOperation = Schema.Schema.Type<typeof HuggingFaceHubOperation>

const HubContext = {
  operation: HuggingFaceHubOperation,
  repository: Schema.OptionFromUndefinedOr(HuggingFaceRepositoryId),
}

export class HuggingFaceAuthenticationError extends Schema.TaggedError<HuggingFaceAuthenticationError>()("HuggingFaceAuthenticationError", HubContext) {}
export class HuggingFaceAccessDeniedError extends Schema.TaggedError<HuggingFaceAccessDeniedError>()("HuggingFaceAccessDeniedError", HubContext) {}
export class HuggingFaceNotFoundError extends Schema.TaggedError<HuggingFaceNotFoundError>()("HuggingFaceNotFoundError", {
  ...HubContext,
  revision: Schema.OptionFromUndefinedOr(HuggingFaceRevision),
  path: Schema.OptionFromUndefinedOr(HuggingFaceFilePath),
}) {}
export class HuggingFaceRateLimitedError extends Schema.TaggedError<HuggingFaceRateLimitedError>()("HuggingFaceRateLimitedError", HubContext) {}
export class HuggingFaceUnavailableError extends Schema.TaggedError<HuggingFaceUnavailableError>()("HuggingFaceUnavailableError", {
  ...HubContext,
  status: Schema.OptionFromUndefinedOr(Schema.NonNegativeInt),
  diagnostic: Schema.String,
}) {}
export class HuggingFaceInvalidResponseError extends Schema.TaggedError<HuggingFaceInvalidResponseError>()("HuggingFaceInvalidResponseError", {
  ...HubContext,
  diagnostic: Schema.String,
}) {}
export class HuggingFaceInvalidRequestError extends Schema.TaggedError<HuggingFaceInvalidRequestError>()("HuggingFaceInvalidRequestError", {
  operation: HuggingFaceHubOperation,
  diagnostic: Schema.String,
}) {}

export const HuggingFaceArtifactInvalidReason = Schema.Literal("empty-files", "duplicate-file", "unsafe-path", "missing-file", "invalid-relationship", "invalid-content-identity", "invalid-artifact", "artifact-id-mismatch")
export type HuggingFaceArtifactInvalidReason = Schema.Schema.Type<typeof HuggingFaceArtifactInvalidReason>
export class HuggingFaceArtifactInvalidError extends Schema.TaggedError<HuggingFaceArtifactInvalidError>()("HuggingFaceArtifactInvalidError", {
  repository: HuggingFaceRepositoryId,
  reason: HuggingFaceArtifactInvalidReason,
  path: Schema.OptionFromUndefinedOr(Schema.String),
}) {}

export const HuggingFaceHubError = Schema.Union(
  HuggingFaceAuthenticationError,
  HuggingFaceAccessDeniedError,
  HuggingFaceNotFoundError,
  HuggingFaceRateLimitedError,
  HuggingFaceUnavailableError,
  HuggingFaceInvalidResponseError,
  HuggingFaceInvalidRequestError,
  HuggingFaceArtifactInvalidError,
)
export type HuggingFaceHubError = Schema.Schema.Type<typeof HuggingFaceHubError>

export class StorageCapacityError extends Schema.TaggedError<StorageCapacityError>()("StorageCapacityError", {
  path: Schema.String,
  diagnostic: Schema.String,
}) {}
export class HuggingFaceInsufficientSpaceError extends Schema.TaggedError<HuggingFaceInsufficientSpaceError>()("HuggingFaceInsufficientSpaceError", {
  requiredBytes: Schema.NonNegativeInt,
  availableBytes: Schema.NonNegativeInt,
}) {}
export class HuggingFaceCacheError extends Schema.TaggedError<HuggingFaceCacheError>()("HuggingFaceCacheError", {
  operation: Schema.Literal("prepare", "inspect", "cleanup"),
  path: Schema.String,
  diagnostic: Schema.String,
}) {}
export class HuggingFaceSizeMismatchError extends Schema.TaggedError<HuggingFaceSizeMismatchError>()("HuggingFaceSizeMismatchError", {
  path: HuggingFaceFilePath,
  expectedBytes: Schema.NonNegativeInt,
  actualBytes: Schema.NonNegativeInt,
}) {}
export class HuggingFaceDigestMismatchError extends Schema.TaggedError<HuggingFaceDigestMismatchError>()("HuggingFaceDigestMismatchError", {
  path: HuggingFaceFilePath,
}) {}
export class HuggingFaceManifestPublicationError extends Schema.TaggedError<HuggingFaceManifestPublicationError>()("HuggingFaceManifestPublicationError", {
  path: Schema.String,
  diagnostic: Schema.String,
}) {}

export const HuggingFaceDownloadError = Schema.Union(
  HuggingFaceAuthenticationError,
  HuggingFaceAccessDeniedError,
  HuggingFaceNotFoundError,
  HuggingFaceRateLimitedError,
  HuggingFaceUnavailableError,
  HuggingFaceInvalidResponseError,
  HuggingFaceInvalidRequestError,
  HuggingFaceArtifactInvalidError,
  StorageCapacityError,
  HuggingFaceInsufficientSpaceError,
  HuggingFaceCacheError,
  HuggingFaceSizeMismatchError,
  HuggingFaceDigestMismatchError,
  HuggingFaceManifestPublicationError,
)
export type HuggingFaceDownloadError = Schema.Schema.Type<typeof HuggingFaceDownloadError>
