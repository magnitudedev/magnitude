import { Schema } from "effect"
import { ModelFileId, ModelFileRole, Sha256Digest, SourceFileRelationshipKind } from "../model-files"
import { DownloadableArtifactId, HuggingFaceCommitId, HuggingFaceRepositoryId, ModelTransferId } from "./identity"
import {
  TransferFailureOperation,
  TransferFailureReason,
} from "./transfer-contracts"

export const SafeRelativePath = Schema.String.pipe(Schema.maxLength(1024), Schema.filter((value) => value.length > 0 && !value.startsWith("/") && !value.startsWith("\\") && !value.includes("\0") && !value.split(/[\\/]/).includes(".."), { message: () => "Expected a contained relative path" }))
const VerifiedFile = Schema.Struct({ path: SafeRelativePath, role: ModelFileRole, shardIndex: Schema.OptionFromUndefinedOr(Schema.NonNegativeInt), sizeBytes: Schema.NonNegativeInt, sha256: Sha256Digest })
const Relationship = Schema.Struct({ kind: SourceFileRelationshipKind, fromPath: SafeRelativePath, toPath: SafeRelativePath })
const Plan = Schema.Struct({ artifactId: DownloadableArtifactId, repository: HuggingFaceRepositoryId, commit: HuggingFaceCommitId, files: Schema.Array(VerifiedFile), relationships: Schema.Array(Relationship), totalBytes: Schema.NonNegativeInt })
const Failure = Schema.Struct({ operation: TransferFailureOperation, reason: TransferFailureReason, path: Schema.OptionFromUndefinedOr(Schema.String), status: Schema.OptionFromUndefinedOr(Schema.Int) })
const Status = Schema.Union(
  Schema.TaggedStruct("Planned", {}),
  Schema.TaggedStruct("CheckingSpace", {}),
  Schema.TaggedStruct("Downloading", { currentFile: Schema.String }),
  Schema.TaggedStruct("Verifying", { currentFile: Schema.String }),
  Schema.TaggedStruct("Publishing", {}),
  Schema.TaggedStruct("Ready", { modelFileId: ModelFileId }),
  Schema.TaggedStruct("Paused", {}),
  Schema.TaggedStruct("Failed", { failure: Failure }),
  Schema.TaggedStruct("Cancelled", {}),
)
const Snapshot = Schema.Struct({ id: ModelTransferId, artifactId: DownloadableArtifactId, status: Status, completedBytes: Schema.NonNegativeInt, totalBytes: Schema.NonNegativeInt })
const PersistedTransfer = Schema.Struct({ version: Schema.Literal(1), plan: Plan, snapshot: Snapshot })
export const PersistedTransferJson = Schema.parseJson(PersistedTransfer, { space: 2 })
