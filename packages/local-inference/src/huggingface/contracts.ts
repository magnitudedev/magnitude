import { Context, Option, Redacted, Schema, type Effect, type Stream } from "effect"
import {
  ModelFileId,
  ModelFileRole,
  ModelFileSourceId,
  Sha256Digest,
  SourceFileRelationshipKind,
} from "../model-files"
import {
  HuggingFaceArtifactId,
  HuggingFaceCommitId,
  HuggingFaceFilePath,
  HuggingFaceObjectId,
  HuggingFaceRepositoryId,
  HuggingFaceRevision,
  HuggingFaceXetHash,
} from "./identity"
import type { HuggingFaceDownloadError, HuggingFaceHubError, StorageCapacityError } from "./errors"

export interface HuggingFaceConnectionOptions {
  readonly hubUrl: Option.Option<URL>
  readonly token: Option.Option<Redacted.Redacted<string>>
  readonly fetch: Option.Option<typeof fetch>
}

export interface HuggingFaceManagedStoreOptions {
  readonly cacheRoot: string
  readonly installationRoot: string
  readonly sourceId: ModelFileSourceId
}

export const HuggingFaceGating = Schema.Union(
  Schema.Literal(false),
  Schema.Literal("auto", "manual"),
)
export type HuggingFaceGating = Schema.Schema.Type<typeof HuggingFaceGating>

export const HuggingFaceModelSort = Schema.Literal("createdAt", "downloads", "likes", "lastModified", "likes30d", "trendingScore", "num_parameters", "mainSize", "id")
export type HuggingFaceModelSort = Schema.Schema.Type<typeof HuggingFaceModelSort>

export const HuggingFaceSearchRequest = Schema.Struct({
  query: Schema.String,
  owner: Schema.OptionFromUndefinedOr(Schema.String),
  tags: Schema.Array(Schema.String),
  apps: Schema.Array(Schema.String),
  sort: Schema.OptionFromUndefinedOr(HuggingFaceModelSort),
  limit: Schema.OptionFromUndefinedOr(Schema.NonNegativeInt),
})
export type HuggingFaceSearchRequest = Schema.Schema.Type<typeof HuggingFaceSearchRequest>

export const HuggingFaceModelSummary = Schema.Struct({
  repository: HuggingFaceRepositoryId,
  private: Schema.Boolean,
  gated: HuggingFaceGating,
  downloads: Schema.NonNegativeInt,
  likes: Schema.NonNegativeInt,
  updatedAt: Schema.DateFromString,
  tags: Schema.Array(Schema.String),
})
export type HuggingFaceModelSummary = Schema.Schema.Type<typeof HuggingFaceModelSummary>

export const HuggingFaceArtifactFileRequest = Schema.Struct({
  path: Schema.String,
  role: ModelFileRole,
  shardIndex: Schema.OptionFromUndefinedOr(Schema.NonNegativeInt),
})
export type HuggingFaceArtifactFileRequest = Schema.Schema.Type<typeof HuggingFaceArtifactFileRequest>

export const HuggingFaceArtifactRelationshipRequest = Schema.Struct({
  kind: SourceFileRelationshipKind,
  fromPath: Schema.String,
  toPath: Schema.String,
})
export type HuggingFaceArtifactRelationshipRequest = Schema.Schema.Type<typeof HuggingFaceArtifactRelationshipRequest>

export const HuggingFaceArtifactRequest = Schema.Struct({
  repository: HuggingFaceRepositoryId,
  revision: HuggingFaceRevision,
  files: Schema.Array(HuggingFaceArtifactFileRequest),
  relationships: Schema.Array(HuggingFaceArtifactRelationshipRequest),
})
export type HuggingFaceArtifactRequest = Schema.Schema.Type<typeof HuggingFaceArtifactRequest>

export class HuggingFaceLfsContent extends Schema.TaggedClass<HuggingFaceLfsContent>("HuggingFaceLfsContent")("LfsSha256", {
  sha256: Sha256Digest,
}) {}
export class HuggingFaceXetContent extends Schema.TaggedClass<HuggingFaceXetContent>("HuggingFaceXetContent")("Xet", {
  hash: HuggingFaceXetHash,
}) {}
export class HuggingFaceGitContent extends Schema.TaggedClass<HuggingFaceGitContent>("HuggingFaceGitContent")("Git", {
  oid: HuggingFaceObjectId,
}) {}

export const HuggingFaceRemoteContentIdentity = Schema.Union(HuggingFaceLfsContent, HuggingFaceXetContent, HuggingFaceGitContent)
export type HuggingFaceRemoteContentIdentity = Schema.Schema.Type<typeof HuggingFaceRemoteContentIdentity>

export const HuggingFaceArtifactFile = Schema.Struct({
  path: HuggingFaceFilePath,
  role: ModelFileRole,
  shardIndex: Schema.OptionFromUndefinedOr(Schema.NonNegativeInt),
  sizeBytes: Schema.NonNegativeInt,
  content: HuggingFaceRemoteContentIdentity,
})
export type HuggingFaceArtifactFile = Schema.Schema.Type<typeof HuggingFaceArtifactFile>

export const HuggingFaceArtifactRelationship = Schema.Struct({
  kind: SourceFileRelationshipKind,
  fromPath: HuggingFaceFilePath,
  toPath: HuggingFaceFilePath,
})
export type HuggingFaceArtifactRelationship = Schema.Schema.Type<typeof HuggingFaceArtifactRelationship>

const HuggingFaceArtifactStruct = Schema.Struct({
  id: HuggingFaceArtifactId,
  repository: HuggingFaceRepositoryId,
  requestedRevision: HuggingFaceRevision,
  commit: HuggingFaceCommitId,
  files: Schema.Array(HuggingFaceArtifactFile),
  relationships: Schema.Array(HuggingFaceArtifactRelationship),
  totalBytes: Schema.NonNegativeInt,
})

export const HuggingFaceArtifact = HuggingFaceArtifactStruct.pipe(
  Schema.filter((artifact) => artifact.files.length > 0, { message: () => "An artifact must contain at least one file" }),
  Schema.filter((artifact) => new Set(artifact.files.map(({ path }) => path)).size === artifact.files.length, { message: () => "Artifact file paths must be unique" }),
  Schema.filter((artifact) => artifact.totalBytes === artifact.files.reduce((sum, file) => sum + file.sizeBytes, 0), { message: () => "Artifact totalBytes must equal the sum of its files" }),
  Schema.filter((artifact) => {
    const paths = new Set(artifact.files.map(({ path }) => path))
    return artifact.relationships.every(({ fromPath, toPath }) => paths.has(fromPath) && paths.has(toPath))
  }, { message: () => "Artifact relationships must reference selected files" }),
)
export type HuggingFaceArtifact = Schema.Schema.Type<typeof HuggingFaceArtifact>

export const DownloadByteProgress = Schema.Struct({
  completedBytes: Schema.NonNegativeInt,
  totalBytes: Schema.NonNegativeInt,
}).pipe(Schema.filter(({ completedBytes, totalBytes }) => completedBytes <= totalBytes, { message: () => "completedBytes must not exceed totalBytes" }))
export type DownloadByteProgress = Schema.Schema.Type<typeof DownloadByteProgress>

export const DownloadFileProgress = Schema.Struct({
  path: HuggingFaceFilePath,
  completedBytes: Schema.NonNegativeInt,
  totalBytes: Schema.NonNegativeInt,
}).pipe(Schema.filter(({ completedBytes, totalBytes }) => completedBytes <= totalBytes, { message: () => "completedBytes must not exceed totalBytes" }))
export type DownloadFileProgress = Schema.Schema.Type<typeof DownloadFileProgress>

export class DownloadCheckingSpace extends Schema.TaggedClass<DownloadCheckingSpace>("DownloadCheckingSpace")("CheckingSpace", {
  artifactId: HuggingFaceArtifactId,
  requiredBytes: Schema.NonNegativeInt,
  availableBytes: Schema.NonNegativeInt,
  aggregate: DownloadByteProgress,
}) {}
export class DownloadingProgress extends Schema.TaggedClass<DownloadingProgress>("DownloadingProgress")("Downloading", {
  artifactId: HuggingFaceArtifactId,
  file: DownloadFileProgress,
  aggregate: DownloadByteProgress,
}) {}
export class DownloadVerifying extends Schema.TaggedClass<DownloadVerifying>("DownloadVerifying")("Verifying", {
  artifactId: HuggingFaceArtifactId,
  path: HuggingFaceFilePath,
  aggregate: DownloadByteProgress,
}) {}
export class DownloadReady extends Schema.TaggedClass<DownloadReady>("DownloadReady")("Ready", {
  artifactId: HuggingFaceArtifactId,
  modelFileId: ModelFileId,
  aggregate: DownloadByteProgress,
}) {}

export const DownloadProgress = Schema.Union(DownloadCheckingSpace, DownloadingProgress, DownloadVerifying, DownloadReady)
export type DownloadProgress = Schema.Schema.Type<typeof DownloadProgress>

export interface HuggingFaceHubApi {
  readonly searchModels: (request: HuggingFaceSearchRequest) => Stream.Stream<HuggingFaceModelSummary, HuggingFaceHubError>
  readonly resolveArtifact: (request: HuggingFaceArtifactRequest) => Effect.Effect<HuggingFaceArtifact, HuggingFaceHubError>
}
export class HuggingFaceHub extends Context.Tag("@magnitudedev/local-inference/HuggingFaceHub")<HuggingFaceHub, HuggingFaceHubApi>() {}

export interface HuggingFaceDownloadApi {
  readonly download: (artifact: HuggingFaceArtifact) => Stream.Stream<DownloadProgress, HuggingFaceDownloadError>
}
export class HuggingFaceDownload extends Context.Tag("@magnitudedev/local-inference/HuggingFaceDownload")<HuggingFaceDownload, HuggingFaceDownloadApi>() {}

export interface StorageCapacityApi {
  readonly availableBytes: (path: string) => Effect.Effect<number, StorageCapacityError>
}
export class StorageCapacity extends Context.Tag("@magnitudedev/local-inference/StorageCapacity")<StorageCapacity, StorageCapacityApi>() {}
