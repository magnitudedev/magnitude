import { Schema } from "effect"

export const HuggingFaceRepositoryId = Schema.String.pipe(
  Schema.pattern(/^[^/\s]+\/[^/\s]+$/),
  Schema.maxLength(512),
  Schema.brand("HuggingFaceRepositoryId"),
)
export type HuggingFaceRepositoryId = Schema.Schema.Type<typeof HuggingFaceRepositoryId>
export const HuggingFaceRevision = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(512), Schema.brand("HuggingFaceRevision"))
export type HuggingFaceRevision = Schema.Schema.Type<typeof HuggingFaceRevision>
export const HuggingFaceCommitId = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{40}$/i), Schema.brand("HuggingFaceCommitId"))
export type HuggingFaceCommitId = Schema.Schema.Type<typeof HuggingFaceCommitId>
export const DownloadableArtifactId = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(1024), Schema.brand("DownloadableArtifactId"))
export type DownloadableArtifactId = Schema.Schema.Type<typeof DownloadableArtifactId>
export const ModelTransferId = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(128), Schema.brand("ModelTransferId"))
export type ModelTransferId = Schema.Schema.Type<typeof ModelTransferId>
export const HuggingFaceObjectId = Schema.String.pipe(Schema.pattern(/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/i), Schema.brand("HuggingFaceObjectId"))
export type HuggingFaceObjectId = Schema.Schema.Type<typeof HuggingFaceObjectId>
