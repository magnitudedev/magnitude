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
export const HuggingFaceArtifactId = Schema.String.pipe(Schema.pattern(/^hf_[a-f0-9]{64}$/), Schema.brand("HuggingFaceArtifactId"))
export type HuggingFaceArtifactId = Schema.Schema.Type<typeof HuggingFaceArtifactId>
export const HuggingFaceFilePath = Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(4096),
  Schema.filter((value) => !value.startsWith("/") && !value.startsWith("\\") && !value.split(/[\\/]/).some((part) => part === "" || part === "." || part === ".."), {
    message: () => "Expected a safe relative Hugging Face file path",
  }),
  Schema.brand("HuggingFaceFilePath"),
)
export type HuggingFaceFilePath = Schema.Schema.Type<typeof HuggingFaceFilePath>
export const HuggingFaceObjectId = Schema.String.pipe(Schema.pattern(/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/i), Schema.brand("HuggingFaceObjectId"))
export type HuggingFaceObjectId = Schema.Schema.Type<typeof HuggingFaceObjectId>
export const HuggingFaceXetHash = Schema.String.pipe(Schema.pattern(/^[a-f0-9]{64}$/i), Schema.brand("HuggingFaceXetHash"))
export type HuggingFaceXetHash = Schema.Schema.Type<typeof HuggingFaceXetHash>
