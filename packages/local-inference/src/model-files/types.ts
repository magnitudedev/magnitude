import { Context, Data, Option, Schema, type Effect, type Stream } from "effect"
import {
  ModelArtifactKey,
  ModelFileFormatId,
  ModelFileId,
  ModelFilePartId,
  ModelFileSourceId,
  ModelFileSourceKind,
  ModelOriginRepositoryId,
  ModelOriginRevisionId,
  Sha256Digest,
  SourceFileKey,
  SourceFileSetId,
} from "./identity"
import type { SchemaIssue } from "../schema-issues"

export const ModelFileRole = Schema.Literal("primary", "shard", "projector", "auxiliary")
export type ModelFileRole = Schema.Schema.Type<typeof ModelFileRole>
export const ModelFileOwnership = Schema.Literal("magnitude", "external")
export type ModelFileOwnership = Schema.Schema.Type<typeof ModelFileOwnership>
export const SourceFileRelationshipKind = Schema.Literal("projector-for")
export type SourceFileRelationshipKind = Schema.Schema.Type<typeof SourceFileRelationshipKind>
export const ModelOriginKind = Schema.Literal("huggingface")
export type ModelOriginKind = Schema.Schema.Type<typeof ModelOriginKind>

export interface SourceFileEntry {
  readonly key: SourceFileKey
  readonly path: string
  readonly relativePath: string
  readonly sizeBytes: number
  readonly modifiedAtMillis: Option.Option<number>
  readonly sha256: Option.Option<Sha256Digest>
  readonly declaredRole: Option.Option<ModelFileRole>
  readonly shardIndex: Option.Option<number>
}

export interface SourceFileRelationship {
  readonly kind: SourceFileRelationshipKind
  readonly from: SourceFileKey
  readonly to: SourceFileKey
}

export interface SourceFileSet {
  readonly id: SourceFileSetId
  readonly artifactKey: Option.Option<ModelArtifactKey>
  readonly sourceId: ModelFileSourceId
  readonly entries: readonly SourceFileEntry[]
  readonly relationships: readonly SourceFileRelationship[]
  readonly origin: Option.Option<{
    readonly kind: ModelOriginKind
    readonly repository: ModelOriginRepositoryId
    readonly revision: Option.Option<ModelOriginRevisionId>
  }>
}

export const SourceDiscoveryIssueCode = Schema.Literal("unreadable", "invalid_manifest", "unsafe_path", "unsupported_layout")
export type SourceDiscoveryIssueCode = Schema.Schema.Type<typeof SourceDiscoveryIssueCode>

export interface SourceDiscoveryIssue {
  readonly sourceId: ModelFileSourceId
  readonly code: SourceDiscoveryIssueCode
  readonly message: string
  readonly sourceKey: Option.Option<SourceFileKey>
}

export type SourceDiscoveryEvent = Data.TaggedEnum<{
  FileSet: { readonly set: SourceFileSet }
  Issue: { readonly issue: SourceDiscoveryIssue }
}>
export const SourceDiscoveryEvent = Data.taggedEnum<SourceDiscoveryEvent>()

export const ModelFileDiscoveryRefresh = Schema.Literal("changed", "full")
export type ModelFileDiscoveryRefresh = Schema.Schema.Type<typeof ModelFileDiscoveryRefresh>

export interface ModelFileDiscoveryRequest {
  readonly refresh: ModelFileDiscoveryRefresh
}

export interface ResolvedSourceFiles {
  readonly set: SourceFileSet
}

export interface ModelFileSource {
  readonly id: ModelFileSourceId
  readonly kind: ModelFileSourceKind
  readonly label: string
  readonly ownership: ModelFileOwnership
  readonly discover: (request: ModelFileDiscoveryRequest) => Stream.Stream<SourceDiscoveryEvent, SourceDiscoveryError>
  readonly resolve: (key: SourceFileSetId) => Effect.Effect<ResolvedSourceFiles, SourceFileSetNotFound | SourceUnavailable>
}

export interface ModelFilePublicationFile {
  readonly key: SourceFileKey
  readonly stagedPath: string
  readonly publishedRelativePath: string
  readonly role: ModelFileRole
  readonly shardIndex: Option.Option<number>
  readonly sizeBytes: number
  readonly sha256: Sha256Digest
}

export interface ModelFilePublication {
  readonly artifactKey: ModelArtifactKey
  readonly files: readonly ModelFilePublicationFile[]
  readonly relationships: readonly SourceFileRelationship[]
  readonly origin: Option.Option<{ readonly kind: ModelOriginKind; readonly repository: ModelOriginRepositoryId; readonly revision: ModelOriginRevisionId }>
}

export interface WritableModelFileSource extends ModelFileSource {
  readonly publish: (publication: ModelFilePublication) => Effect.Effect<ModelFileId, ModelFilePublishError>
}

export interface DeletableModelFileSource extends ModelFileSource {
  readonly remove: (id: ModelFileId) => Effect.Effect<void, ModelFileDeleteError>
}

export type ModelFileSourceRegistration = Data.TaggedEnum<{
  ReadOnly: { readonly source: ModelFileSource }
  Deletable: { readonly source: DeletableModelFileSource }
}>
export const ModelFileSourceRegistration = Data.taggedEnum<ModelFileSourceRegistration>()

export interface ModelFileMetadata {
  readonly name: Option.Option<string>
  readonly architecture: Option.Option<string>
  readonly ggufFileType: Option.Option<number>
  readonly quantization: Option.Option<string>
  readonly trainedContextTokens: Option.Option<number>
  readonly parameterCount: Option.Option<number>
  readonly embeddingLength: Option.Option<number>
  readonly blockCount: Option.Option<number>
  readonly attentionHeadCount: Option.Option<number>
  readonly vocabularySize: Option.Option<number>
  readonly feedForwardLength: Option.Option<number>
  readonly expertCount: Option.Option<number>
  readonly expertUsedCount: Option.Option<number>
  readonly tokenizerModel: Option.Option<string>
  readonly tokenizerPre: Option.Option<string>
  readonly chatTemplate: Option.Option<string>
  readonly baseModelNames: readonly string[]
  readonly baseModelRepositories: readonly string[]
  readonly inputModalities: Option.Option<readonly string[]>
  readonly outputModalities: Option.Option<readonly string[]>
}

export interface InspectedModelPart {
  readonly entry: SourceFileEntry
  readonly role: ModelFileRole
}

export interface InspectedModelArtifact {
  readonly key: ModelArtifactKey
  readonly displayName: string
  readonly parts: readonly InspectedModelPart[]
  readonly metadata: ModelFileMetadata
  readonly warnings: readonly ModelFileWarning[]
}

export const SourceFileEntrySchema = Schema.Struct({
  key: SourceFileKey,
  path: Schema.String,
  relativePath: Schema.String,
  sizeBytes: Schema.NonNegativeInt,
  modifiedAtMillis: Schema.optionalWith(Schema.NonNegativeInt, { as: "Option", exact: true }),
  sha256: Schema.optionalWith(Sha256Digest, { as: "Option", exact: true }),
  declaredRole: Schema.optionalWith(ModelFileRole, { as: "Option", exact: true }),
  shardIndex: Schema.optionalWith(Schema.NonNegativeInt, { as: "Option", exact: true }),
})

export const SourceFileSetSchema = Schema.Struct({
  id: SourceFileSetId,
  artifactKey: Schema.optionalWith(ModelArtifactKey, { as: "Option", exact: true }),
  sourceId: ModelFileSourceId,
  entries: Schema.Array(SourceFileEntrySchema),
  relationships: Schema.Array(Schema.Struct({
    kind: SourceFileRelationshipKind,
    from: SourceFileKey,
    to: SourceFileKey,
  })),
  origin: Schema.optionalWith(Schema.Struct({
    kind: ModelOriginKind,
    repository: ModelOriginRepositoryId,
    revision: Schema.optionalWith(ModelOriginRevisionId, { as: "Option", exact: true }),
  }), { as: "Option", exact: true }),
})

export const ModelFileMetadataSchema = Schema.Struct({
  name: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  architecture: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  ggufFileType: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  quantization: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  trainedContextTokens: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  parameterCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  embeddingLength: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  blockCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  attentionHeadCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  vocabularySize: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  feedForwardLength: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  expertCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  expertUsedCount: Schema.optionalWith(Schema.Number, { as: "Option", exact: true }),
  tokenizerModel: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  tokenizerPre: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  chatTemplate: Schema.optionalWith(Schema.String, { as: "Option", exact: true }),
  baseModelNames: Schema.Array(Schema.String),
  baseModelRepositories: Schema.Array(Schema.String),
  inputModalities: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
  outputModalities: Schema.optionalWith(Schema.Array(Schema.String), { as: "Option", exact: true }),
})

export const ModelFileWarningSchema = Schema.Union(
  Schema.TaggedStruct("DuplicatePart", { key: SourceFileKey }),
  Schema.TaggedStruct("RelationshipConflict", { key: SourceFileKey, declaredRole: ModelFileRole }),
  Schema.TaggedStruct("IncompleteSplit", { expectedParts: Schema.NonNegativeInt, observedParts: Schema.NonNegativeInt }),
)

export const InspectedModelArtifactSchema = Schema.Struct({
  key: ModelArtifactKey,
  displayName: Schema.String,
  parts: Schema.Array(Schema.Struct({ entry: SourceFileEntrySchema, role: ModelFileRole })),
  metadata: ModelFileMetadataSchema,
  warnings: Schema.Array(ModelFileWarningSchema),
})

export const ModelArtifactIndexSchema = Schema.Struct({
  capturedAt: Schema.DateFromString,
  sets: Schema.Array(Schema.Struct({
    sourceId: ModelFileSourceId,
    set: SourceFileSetSchema,
    formatId: ModelFileFormatId,
    version: Schema.String,
    artifacts: Schema.Array(InspectedModelArtifactSchema),
  })),
  issues: Schema.Array(Schema.Struct({
    sourceId: ModelFileSourceId,
    code: SourceDiscoveryIssueCode,
    message: Schema.String,
    sourceKey: Schema.optionalWith(SourceFileKey, { as: "Option", exact: true }),
  })),
})
export type ModelArtifactIndex = Schema.Schema.Type<typeof ModelArtifactIndexSchema>

export interface ModelFileFormat {
  readonly id: ModelFileFormatId
  readonly recognize: (file: SourceFileEntry) => Effect.Effect<boolean, ModelFormatError>
  readonly inspect: (set: SourceFileSet) => Effect.Effect<readonly InspectedModelArtifact[], ModelFormatError>
}

export interface ModelFilePart {
  readonly id: ModelFilePartId
  readonly role: ModelFileRole
  readonly sizeBytes: number
  readonly sha256: Option.Option<Sha256Digest>
}

export type ModelFileWarning = Data.TaggedEnum<{
  DuplicatePart: {
    readonly key: SourceFileKey
  }
  RelationshipConflict: {
    readonly key: SourceFileKey
    readonly declaredRole: ModelFileRole
  }
  IncompleteSplit: {
    readonly expectedParts: number
    readonly observedParts: number
  }
}>

export const ModelFileWarning = Data.taggedEnum<ModelFileWarning>()

export interface ModelFileRecord {
  readonly id: ModelFileId
  readonly sourceId: ModelFileSourceId
  readonly displayName: string
  readonly format: ModelFileFormatId
  readonly sizeBytes: number
  readonly files: readonly ModelFilePart[]
  readonly metadata: ModelFileMetadata
  readonly ownership: ModelFileOwnership
  readonly operations: { readonly delete: boolean }
  readonly warnings: readonly ModelFileWarning[]
}

export interface ModelFileVersionPart {
  readonly key: SourceFileKey
  readonly sizeBytes: number
  readonly modifiedAtMillis: Option.Option<number>
}

export interface ModelFileSourceSummary {
  readonly id: ModelFileSourceId
  readonly kind: ModelFileSourceKind
  readonly label: string
  readonly ownership: ModelFileOwnership
}

export interface ResolvedModelFiles {
  readonly record: ModelFileRecord
  readonly primaryPath: string
  readonly shardPaths: readonly string[]
  readonly projectorPath: Option.Option<string>
  readonly auxiliaryPaths: readonly string[]
  readonly version: readonly ModelFileVersionPart[]
}

export interface ModelFileSnapshot {
  readonly records: readonly ModelFileRecord[]
  readonly issues: readonly SourceDiscoveryIssue[]
  readonly capturedAt: Date
}

export const ModelFileRefresh = Schema.Literal("cached", "changed", "full")
export type ModelFileRefresh = Schema.Schema.Type<typeof ModelFileRefresh>

export interface ModelFileRegistryApi {
  readonly inspect: (refresh: ModelFileRefresh) => Effect.Effect<ModelFileSnapshot>
  readonly get: (id: ModelFileId) => Effect.Effect<ModelFileRecord, ModelFileNotFound>
  readonly resolve: (id: ModelFileId) => Effect.Effect<ResolvedModelFiles, ModelFileResolveError>
  readonly remove: (id: ModelFileId) => Effect.Effect<void, ModelFileDeleteError>
  readonly artifactIndex: Effect.Effect<ModelArtifactIndex>
  readonly changes: Stream.Stream<void>
}
export class ModelFileRegistry extends Context.Tag("@magnitudedev/local-inference/ModelFileRegistry")<ModelFileRegistry, ModelFileRegistryApi>() {}

export const FileSystemFailureReason = Schema.Literal("not-found", "already-exists", "permission-denied", "invalid-data", "bad-argument", "bad-resource", "busy", "timed-out", "unexpected-eof", "system-unknown", "would-block", "write-zero")
export type FileSystemFailureReason = Schema.Schema.Type<typeof FileSystemFailureReason>
export const SourceDiscoveryOperation = Schema.Literal("read-root", "read-directory", "inspect-entry", "resolve-link")
export type SourceDiscoveryOperation = Schema.Schema.Type<typeof SourceDiscoveryOperation>
export const SourceUnavailableReason = Schema.Literal("not-found", "unreadable")
export type SourceUnavailableReason = Schema.Schema.Type<typeof SourceUnavailableReason>
export const ModelFilePublishOperation = Schema.Literal("validate", "stage", "verify", "commit")
export type ModelFilePublishOperation = Schema.Schema.Type<typeof ModelFilePublishOperation>
export const ModelFilePublishReason = Schema.Union(Schema.Literal("empty", "primary-count", "duplicate-key", "invalid-relationship", "unsafe-path", "size-mismatch", "digest-mismatch"), FileSystemFailureReason)
export type ModelFilePublishReason = Schema.Schema.Type<typeof ModelFilePublishReason>
export const ModelFileDeleteReason = Schema.Union(Schema.Literal("not-found", "read-only", "source-unavailable"), FileSystemFailureReason)
export type ModelFileDeleteReason = Schema.Schema.Type<typeof ModelFileDeleteReason>
export const ModelFormatOperation = Schema.Literal("recognize", "read", "decode", "assemble")
export type ModelFormatOperation = Schema.Schema.Type<typeof ModelFormatOperation>
export const ModelFormatFailureReason = Schema.Literal("unreadable", "invalid-header", "unsupported-version", "invalid-metadata", "foreign-library-failure")
export type ModelFormatFailureReason = Schema.Schema.Type<typeof ModelFormatFailureReason>
export const ModelFileResolveFailureReason = Schema.Literal("not-found", "source-unavailable", "part-missing", "changed", "unreadable")
export type ModelFileResolveFailureReason = Schema.Schema.Type<typeof ModelFileResolveFailureReason>

export class SourceDiscoveryError extends Data.TaggedError("SourceDiscoveryError")<{ readonly sourceId: ModelFileSourceId; readonly operation: SourceDiscoveryOperation; readonly reason: FileSystemFailureReason; readonly path: string }> {}
export class SourceUnavailable extends Data.TaggedError("SourceUnavailable")<{ readonly sourceId: ModelFileSourceId; readonly setId: SourceFileSetId; readonly reason: SourceUnavailableReason }> {}
export class SourceFileSetNotFound extends Data.TaggedError("SourceFileSetNotFound")<{ readonly id: SourceFileSetId }> {}
export class ModelFileNotFound extends Data.TaggedError("ModelFileNotFound")<{ readonly id: ModelFileId }> {}
export class ModelFilePublishError extends Data.TaggedError("ModelFilePublishError")<{ readonly sourceId: ModelFileSourceId; readonly artifactKey: ModelArtifactKey; readonly operation: ModelFilePublishOperation; readonly reason: ModelFilePublishReason; readonly path: Option.Option<string> }> {}
export class ModelFileDeleteError extends Data.TaggedError("ModelFileDeleteError")<{ readonly id: ModelFileId; readonly reason: ModelFileDeleteReason }> {}
export class ModelFormatError extends Data.TaggedError("ModelFormatError")<{ readonly format: ModelFileFormatId; readonly file: SourceFileKey; readonly operation: ModelFormatOperation; readonly reason: ModelFormatFailureReason; readonly diagnostic: Option.Option<string>; readonly issues: readonly SchemaIssue[] }> {}
export class ModelFileResolveError extends Data.TaggedError("ModelFileResolveError")<{ readonly id: ModelFileId; readonly reason: ModelFileResolveFailureReason; readonly part: Option.Option<SourceFileKey> }> {}
