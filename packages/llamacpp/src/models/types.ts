import { Schema } from "effect"

export const LocalModelSource = Schema.Union(
  Schema.TaggedStruct("hf-cache", {
    repoId: Schema.String,
    commit: Schema.String,
  }),
  Schema.TaggedStruct("user-dir", {
    dir: Schema.String,
  }),
)
export type LocalModelSource = Schema.Schema.Type<typeof LocalModelSource>

/** Internal filesystem discovery value. Paths never cross the package boundary. */
export const LocalModelInfo = Schema.Struct({
  id: Schema.String,
  displayName: Schema.String,
  filePath: Schema.String,
  shardPaths: Schema.optional(Schema.Array(Schema.String)),
  mmprojPath: Schema.optional(Schema.String),
  architecture: Schema.optional(Schema.String),
  quantization: Schema.optional(Schema.String),
  contextLength: Schema.optional(Schema.Number),
  fileSizeBytes: Schema.Number,
  parameterCount: Schema.optional(Schema.Number),
  hiddenSize: Schema.optional(Schema.Number),
  layerCount: Schema.optional(Schema.Number),
  headCount: Schema.optional(Schema.Number),
  vocabSize: Schema.optional(Schema.Number),
  feedForwardLength: Schema.optional(Schema.Number),
  expertCount: Schema.optional(Schema.Number),
  expertUsedCount: Schema.optional(Schema.Number),
  tokenizerModel: Schema.optional(Schema.String),
  tokenizerPre: Schema.optional(Schema.String),
  chatTemplate: Schema.optional(Schema.String),
  chatTemplatePresent: Schema.Boolean,
  vision: Schema.Boolean,
  audio: Schema.Boolean,
  moe: Schema.Boolean,
  source: LocalModelSource,
  repoId: Schema.optional(Schema.String),
  commit: Schema.optional(Schema.String),
  baseModelNames: Schema.optional(Schema.Array(Schema.String)),
})
export type LocalModelInfo = Schema.Schema.Type<typeof LocalModelInfo>

export const ExpandedGgufMetadata = Schema.Struct({
  generalName: Schema.optional(Schema.String),
  generalBasename: Schema.optional(Schema.String),
  generalSizeLabel: Schema.optional(Schema.String),
  generalFinetune: Schema.optional(Schema.String),
  generalVersion: Schema.optional(Schema.String),
  architecture: Schema.optional(Schema.String),
  quantization: Schema.optional(Schema.String),
  contextLength: Schema.optional(Schema.Number),
  hiddenSize: Schema.optional(Schema.Number),
  layerCount: Schema.optional(Schema.Number),
  headCount: Schema.optional(Schema.Number),
  vocabSize: Schema.optional(Schema.Number),
  expertCount: Schema.optional(Schema.Number),
  expertUsedCount: Schema.optional(Schema.Number),
  feedForwardLength: Schema.optional(Schema.Number),
  parameterCount: Schema.optional(Schema.Number),
  tokenizerModel: Schema.optional(Schema.String),
  tokenizerPre: Schema.optional(Schema.String),
  chatTemplate: Schema.optional(Schema.String),
  chatTemplatePresent: Schema.Boolean,
  baseModelNames: Schema.Array(Schema.String),
  baseModelRepositories: Schema.Array(Schema.String),
})
export type ExpandedGgufMetadata = Schema.Schema.Type<typeof ExpandedGgufMetadata>

export const ShardGroup = Schema.Struct({
  prefix: Schema.String,
  total: Schema.Number,
  shards: Schema.Array(Schema.String),
  primaryPath: Schema.String,
})
export type ShardGroup = Schema.Schema.Type<typeof ShardGroup>
