import { GGMLFileQuantizationType } from "@huggingface/tasks"
import { Option, pipe, Schema } from "effect"
import type { ModelFileMetadata } from "./types"
import type { GgufTypedMetadata } from "./gguf-schema"

export const GgufKey = {
  GeneralArchitecture: "general.architecture",
  GeneralFileType: "general.file_type",
  GeneralName: "general.name",
  GeneralType: "general.type",
  SplitIndex: "split.no",
  SplitCount: "split.count",
  TokenizerModel: "tokenizer.ggml.model",
  TokenizerPre: "tokenizer.ggml.pre",
  ChatTemplate: "tokenizer.chat_template",
} as const

export const GgufArchitectureField = {
  ContextLength: "context_length",
  EmbeddingLength: "embedding_length",
  BlockCount: "block_count",
  AttentionHeadCount: "attention.head_count",
  VocabularySize: "vocab_size",
  FeedForwardLength: "feed_forward_length",
  ExpertCount: "expert_count",
  ExpertUsedCount: "expert_used_count",
} as const
export type GgufArchitectureField = typeof GgufArchitectureField[keyof typeof GgufArchitectureField]
export const GgufMetadataNamespace = Schema.Literal("general.base_model")
export type GgufMetadataNamespace = Schema.Schema.Type<typeof GgufMetadataNamespace>
export const GgufMetadataField = Schema.Literal("name", "repo")
export type GgufMetadataField = Schema.Schema.Type<typeof GgufMetadataField>
type FixedGgufKey = typeof GgufKey[keyof typeof GgufKey]
export type GgufMetadataKey = FixedGgufKey | `${string}.${GgufArchitectureField}`

const MetadataNumber = Schema.Union(Schema.Number, Schema.BigIntFromSelf)

const decodeString = Schema.decodeUnknownOption(Schema.String)
const decodeNumber = (value: unknown): Option.Option<number> => pipe(
  Schema.decodeUnknownOption(MetadataNumber)(value),
  Option.map((number) => typeof number === "bigint" ? Number(number) : number),
  Option.filter(Number.isFinite),
)

export class GgufMetadata {
  readonly architecture: Option.Option<string>

  constructor(private readonly values: GgufTypedMetadata) {
    this.architecture = this.string(GgufKey.GeneralArchitecture)
  }

  string(key: GgufMetadataKey): Option.Option<string> {
    return pipe(
      Option.fromNullable(this.values[key]),
      Option.flatMap((entry) => decodeString(entry.value)),
    )
  }

  finiteNumber(key: GgufMetadataKey): Option.Option<number> {
    return pipe(
      Option.fromNullable(this.values[key]),
      Option.flatMap((entry) => decodeNumber(entry.value)),
    )
  }

  architectureNumber(field: GgufArchitectureField): Option.Option<number> {
    return pipe(
      this.architecture,
      Option.flatMap((architecture) => this.finiteNumber(`${architecture}.${field}`)),
    )
  }

  indexedStrings(namespace: GgufMetadataNamespace, field: GgufMetadataField): readonly string[] {
    const prefix = `${namespace}.`
    const suffix = `.${field}`
    const entries: Array<readonly [number, string]> = []
    for (const [key, entry] of Object.entries(this.values)) {
      if (!key.startsWith(prefix) || !key.endsWith(suffix)) continue
      const indexText = key.slice(prefix.length, -suffix.length)
      const index = Schema.decodeUnknownOption(Schema.NumberFromString.pipe(Schema.int(), Schema.nonNegative()))(indexText)
      const value = decodeString(entry.value)
      if (Option.isSome(index) && Option.isSome(value)) entries.push([index.value, value.value])
    }
    return entries.sort(([left], [right]) => left - right).map(([, value]) => value)
  }

  quantization(): Option.Option<string> {
    return pipe(
      this.finiteNumber(GgufKey.GeneralFileType),
      Option.flatMap((fileType) => Schema.decodeUnknownOption(Schema.String)(GGMLFileQuantizationType[fileType])),
    )
  }
}

export const normalizeParameterCount = (
  value: Option.Option<number | bigint>,
): Option.Option<number> => pipe(
  value,
  Option.map((number) => typeof number === "bigint" ? Number(number) : number),
  Option.filter(Number.isFinite),
)

export const projectGgufMetadata = (
  metadata: GgufMetadata,
  parameterCount: Option.Option<number>,
): ModelFileMetadata => ({
  name: metadata.string(GgufKey.GeneralName),
  architecture: metadata.architecture,
  ggufFileType: metadata.finiteNumber(GgufKey.GeneralFileType),
  quantization: metadata.quantization(),
  trainedContextTokens: metadata.architectureNumber(GgufArchitectureField.ContextLength),
  parameterCount,
  embeddingLength: metadata.architectureNumber(GgufArchitectureField.EmbeddingLength),
  blockCount: metadata.architectureNumber(GgufArchitectureField.BlockCount),
  attentionHeadCount: metadata.architectureNumber(GgufArchitectureField.AttentionHeadCount),
  vocabularySize: metadata.architectureNumber(GgufArchitectureField.VocabularySize),
  feedForwardLength: metadata.architectureNumber(GgufArchitectureField.FeedForwardLength),
  expertCount: metadata.architectureNumber(GgufArchitectureField.ExpertCount),
  expertUsedCount: metadata.architectureNumber(GgufArchitectureField.ExpertUsedCount),
  tokenizerModel: metadata.string(GgufKey.TokenizerModel),
  tokenizerPre: metadata.string(GgufKey.TokenizerPre),
  chatTemplate: metadata.string(GgufKey.ChatTemplate),
  baseModelNames: metadata.indexedStrings("general.base_model", "name"),
  baseModelRepositories: metadata.indexedStrings("general.base_model", "repo"),
  inputModalities: Option.none(),
  outputModalities: Option.none(),
})
