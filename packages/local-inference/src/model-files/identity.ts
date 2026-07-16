import { createHash } from "node:crypto"
import { Schema } from "effect"

const bounded = <const Brand extends string>(brand: Brand) => Schema.String.pipe(
  Schema.minLength(1),
  Schema.maxLength(1024),
  Schema.brand(brand),
)

export const ModelFileSourceId = bounded("ModelFileSourceId")
export type ModelFileSourceId = Schema.Schema.Type<typeof ModelFileSourceId>
export const ModelFileSourceKind = bounded("ModelFileSourceKind")
export type ModelFileSourceKind = Schema.Schema.Type<typeof ModelFileSourceKind>
export const ModelFileFormatId = bounded("ModelFileFormatId")
export type ModelFileFormatId = Schema.Schema.Type<typeof ModelFileFormatId>
export const SourceFileKey = bounded("SourceFileKey")
export type SourceFileKey = Schema.Schema.Type<typeof SourceFileKey>
export const SourceFileSetId = bounded("SourceFileSetId")
export type SourceFileSetId = Schema.Schema.Type<typeof SourceFileSetId>
export const ModelArtifactKey = bounded("ModelArtifactKey")
export type ModelArtifactKey = Schema.Schema.Type<typeof ModelArtifactKey>
export const ModelFileId = bounded("ModelFileId")
export type ModelFileId = Schema.Schema.Type<typeof ModelFileId>
export const ModelFilePartId = bounded("ModelFilePartId")
export type ModelFilePartId = Schema.Schema.Type<typeof ModelFilePartId>
export const ModelFileVersionId = bounded("ModelFileVersionId")
export type ModelFileVersionId = Schema.Schema.Type<typeof ModelFileVersionId>
export const ModelOriginRepositoryId = bounded("ModelOriginRepositoryId")
export type ModelOriginRepositoryId = Schema.Schema.Type<typeof ModelOriginRepositoryId>
export const ModelOriginRevisionId = bounded("ModelOriginRevisionId")
export type ModelOriginRevisionId = Schema.Schema.Type<typeof ModelOriginRevisionId>
export const Sha256Digest = Schema.String.pipe(
  Schema.pattern(/^[a-f0-9]{64}$/),
  Schema.brand("Sha256Digest"),
)
export type Sha256Digest = Schema.Schema.Type<typeof Sha256Digest>

const digest = (namespace: string, value: string): string =>
  `${namespace}_${createHash("sha256").update(value).digest("hex")}`

export const makeSourceFileKey = (sourceIdentity: string): SourceFileKey => SourceFileKey.make(digest("source", sourceIdentity))
export const makeSourceFileSetId = (sourceIdentity: string): SourceFileSetId => SourceFileSetId.make(digest("set", sourceIdentity))
export const makeModelFileId = (sourceId: ModelFileSourceId, key: ModelArtifactKey): ModelFileId =>
  ModelFileId.make(digest("mf", `${sourceId}\0${key}`))
export const makeModelFilePartId = (sourceId: ModelFileSourceId, key: SourceFileKey): ModelFilePartId =>
  ModelFilePartId.make(digest("part", `${sourceId}\0${key}`))
