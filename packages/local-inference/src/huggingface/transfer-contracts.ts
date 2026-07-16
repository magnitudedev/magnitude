import { Context, Data, Effect, Option, Schema, Stream } from "effect"
import {
  FileSystemFailureReason,
  ModelFileId,
  ModelFileRole,
  Sha256Digest,
  SourceFileRelationshipKind,
  type WritableModelFileSource,
} from "../model-files"
import {
  type DownloadableArtifactId as ArtifactId,
  type HuggingFaceCommitId as CommitId,
  type HuggingFaceRepositoryId as RepositoryId,
  type HuggingFaceRevision as Revision,
  type ModelTransferId as TransferId,
} from "./identity"
import type { HuggingFaceHubClientApi } from "./hub-client"

export const ModelTransferState = Schema.Literal("Planned", "CheckingSpace", "Downloading", "Verifying", "Publishing", "Ready", "Paused", "Failed", "Cancelled")
export type ModelTransferState = Schema.Schema.Type<typeof ModelTransferState>
export const TransferFailureOperation = Schema.Literal("space", "download", "verify", "publish", "persist")
export type TransferFailureOperation = Schema.Schema.Type<typeof TransferFailureOperation>
export const TransferFailureReason = Schema.Union(Schema.Literal("insufficient-space", "capacity-unavailable", "persistence-failed", "transport", "http-rejected", "range-rejected", "size-mismatch", "digest-mismatch", "destination-rejected", "unsafe-path"), FileSystemFailureReason)
export type TransferFailureReason = Schema.Schema.Type<typeof TransferFailureReason>
export const TransferPlanningOperation = Schema.Literal("validate", "resolve-revision", "list-files")
export type TransferPlanningOperation = Schema.Schema.Type<typeof TransferPlanningOperation>
export const TransferPlanningFailureReason = Schema.Literal("empty-files", "duplicate-file", "unsafe-path", "invalid-relationship", "revision-unavailable", "listing-unavailable", "file-missing", "digest-unavailable")
export type TransferPlanningFailureReason = Schema.Schema.Type<typeof TransferPlanningFailureReason>
export const TransferRegistryOperation = Schema.Literal("restore", "persist")
export type TransferRegistryOperation = Schema.Schema.Type<typeof TransferRegistryOperation>
export const TransferRegistryFailureReason = Schema.Union(Schema.Literal("unreadable", "invalid-record"), FileSystemFailureReason)
export type TransferRegistryFailureReason = Schema.Schema.Type<typeof TransferRegistryFailureReason>
export const TransferStateOperation = Schema.Literal("cancel", "resume")
export type TransferStateOperation = Schema.Schema.Type<typeof TransferStateOperation>

export interface HuggingFaceArtifactFileRequest {
  readonly path: string
  readonly role: Schema.Schema.Type<typeof ModelFileRole>
  readonly shardIndex: Option.Option<number>
}

export interface HuggingFaceArtifactRelationshipRequest {
  readonly kind: Schema.Schema.Type<typeof SourceFileRelationshipKind>
  readonly fromPath: string
  readonly toPath: string
}

export interface HuggingFaceArtifactRequest {
  readonly repository: RepositoryId
  readonly revision: Revision
  readonly files: readonly HuggingFaceArtifactFileRequest[]
  readonly relationships: readonly HuggingFaceArtifactRelationshipRequest[]
}

export interface VerifiedTransferFile {
  readonly path: string
  readonly role: Schema.Schema.Type<typeof ModelFileRole>
  readonly shardIndex: Option.Option<number>
  readonly sizeBytes: number
  readonly sha256: Schema.Schema.Type<typeof Sha256Digest>
}

export interface VerifiedTransferPlan {
  readonly artifactId: ArtifactId
  readonly repository: RepositoryId
  readonly commit: CommitId
  readonly files: readonly VerifiedTransferFile[]
  readonly relationships: readonly HuggingFaceArtifactRelationshipRequest[]
  readonly totalBytes: number
}

export interface TransferFailure {
  readonly operation: TransferFailureOperation
  readonly reason: TransferFailureReason
  readonly path: Option.Option<string>
  readonly status: Option.Option<number>
}
export type ModelTransferStatus = Data.TaggedEnum<{
  Planned: Record<never, never>
  CheckingSpace: Record<never, never>
  Downloading: { readonly currentFile: string }
  Verifying: { readonly currentFile: string }
  Publishing: Record<never, never>
  Ready: { readonly modelFileId: Schema.Schema.Type<typeof ModelFileId> }
  Paused: Record<never, never>
  Failed: { readonly failure: TransferFailure }
  Cancelled: Record<never, never>
}>
export const ModelTransferStatus = Data.taggedEnum<ModelTransferStatus>()
export interface ModelTransferSnapshot {
  readonly id: TransferId
  readonly artifactId: ArtifactId
  readonly status: ModelTransferStatus
  readonly completedBytes: number
  readonly totalBytes: number
}
export class TransferPlanningError extends Data.TaggedError("TransferPlanningError")<{ readonly operation: TransferPlanningOperation; readonly reason: TransferPlanningFailureReason; readonly repository: RepositoryId; readonly path: Option.Option<string> }> {}
export class TransferExecutionError extends Data.TaggedError("TransferExecutionError")<TransferFailure & { readonly transferId: TransferId }> {}
export class TransferRegistryError extends Data.TaggedError("TransferRegistryError")<{ readonly operation: TransferRegistryOperation; readonly reason: TransferRegistryFailureReason; readonly path: string }> {}
export class TransferNotFound extends Data.TaggedError("TransferNotFound")<{ readonly id: TransferId }> {}
export class TransferStateError extends Data.TaggedError("TransferStateError")<{ readonly id: TransferId; readonly operation: TransferStateOperation; readonly state: ModelTransferState }> {}
export class StorageCapacityError extends Data.TaggedError("StorageCapacityError")<{ readonly path: string }> {}
export interface StorageCapacityApi {
  readonly availableBytes: (path: string) => Effect.Effect<number, StorageCapacityError>
}

export interface ModelTransferRegistryApi {
  readonly plan: (
    request: HuggingFaceArtifactRequest,
  ) => Effect.Effect<VerifiedTransferPlan, TransferPlanningError>
  readonly start: (
    plan: VerifiedTransferPlan,
  ) => Effect.Effect<TransferId, TransferRegistryError>
  readonly observe: (
    id: TransferId,
  ) => Stream.Stream<ModelTransferSnapshot, TransferNotFound>
  readonly list: Effect.Effect<readonly ModelTransferSnapshot[]>
  readonly cancel: (
    id: TransferId,
  ) => Effect.Effect<void, TransferNotFound | TransferStateError>
  readonly resume: (
    id: TransferId,
  ) => Effect.Effect<void, TransferNotFound | TransferStateError | TransferRegistryError>
  readonly recoveryDiagnostics: Effect.Effect<readonly TransferRegistryError[]>
}
export class ModelTransferRegistry extends Context.Tag("@magnitudedev/local-inference/ModelTransferRegistry")<ModelTransferRegistry, ModelTransferRegistryApi>() {}
export interface ModelTransferRegistryOptions {
  readonly hub: HuggingFaceHubClientApi
  readonly destination: WritableModelFileSource
  readonly capacity: StorageCapacityApi
  readonly stagingRoot: string
  readonly stateRoot: Option.Option<string>
  readonly reserveBytes: number
}
