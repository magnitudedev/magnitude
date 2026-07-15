import { Effect, Option, Schema } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { gguf } from "@huggingface/gguf"
import { GGMLFileQuantizationType } from "@huggingface/tasks"
import type { ExpandedGgufMetadata } from "./types"

// ── Output schema ──

/**
 * Schema for the structured GGUF metadata we extract from the library's flat KV record.
 * All fields are optional except `chatTemplatePresent` and the two arrays (which default to empty).
 */
export const ExpandedGgufMetadataSchema = Schema.Struct({
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
  tokenizerModel: Schema.optional(Schema.String),
  tokenizerPre: Schema.optional(Schema.String),
  chatTemplate: Schema.optional(Schema.String),
  chatTemplatePresent: Schema.Boolean,
  parameterCount: Schema.optional(Schema.Number),
  baseModelNames: Schema.Array(Schema.String),
  baseModelRepositories: Schema.Array(Schema.String),
})

// ── Field extraction from the library's typed metadata ──

type TypedMetadata = Record<string, { value: unknown; type: number; subType?: number }>

/** Decode a string field from the typed metadata record. */
function str(tm: TypedMetadata, key: string): string | undefined {
  const entry = tm[key]
  if (!entry) return undefined
  return typeof entry.value === "string" ? entry.value : undefined
}

/** Decode a number field from the typed metadata record (handles bigint). */
function num(tm: TypedMetadata, key: string): number | undefined {
  const entry = tm[key]
  if (!entry) return undefined
  const v = entry.value
  return typeof v === "number" ? v : typeof v === "bigint" ? Number(v) : undefined
}

/** Extract an indexed array from `general.base_model.{i}.{field}` keys. */
function baseModelArray(tm: TypedMetadata, field: string): string[] {
  const results: string[] = []
  for (let i = 0; ; i++) {
    const value = str(tm, `general.base_model.${i}.${field}`)
    if (!value) break
    results.push(value)
  }
  return results
}

/**
 * Map the flat typed metadata KV record into the `ExpandedGgufMetadata` shape.
 * Resolves architecture-specific key prefixes from `general.architecture`.
 * Maps `general.file_type` to the `GGMLFileQuantizationType` enum name.
 */
function toStructuredMetadata(tm: TypedMetadata): unknown {
  const architecture = str(tm, "general.architecture")
  const prefix = architecture ? `${architecture}.` : ""
  const chatTemplate = str(tm, "tokenizer.chat_template")

  const fileType = num(tm, "general.file_type")
  const quantization = fileType !== undefined
    ? GGMLFileQuantizationType[fileType as GGMLFileQuantizationType]
    : undefined

  return {
    generalName: str(tm, "general.name"),
    generalBasename: str(tm, "general.basename"),
    generalSizeLabel: str(tm, "general.size_label"),
    generalFinetune: str(tm, "general.finetune"),
    generalVersion: str(tm, "general.version"),
    architecture,
    quantization,
    contextLength: num(tm, `${prefix}context_length`),
    hiddenSize: num(tm, `${prefix}embedding_length`),
    layerCount: num(tm, `${prefix}block_count`),
    headCount: num(tm, `${prefix}attention.head_count`),
    vocabSize: num(tm, `${prefix}vocab_size`),
    expertCount: num(tm, `${prefix}expert_count`),
    expertUsedCount: num(tm, `${prefix}expert_used_count`),
    feedForwardLength: num(tm, `${prefix}feed_forward_length`),
    tokenizerModel: str(tm, "tokenizer.ggml.model"),
    tokenizerPre: str(tm, "tokenizer.ggml.pre"),
    chatTemplate,
    chatTemplatePresent: chatTemplate !== undefined,
    baseModelNames: baseModelArray(tm, "name"),
    baseModelRepositories: baseModelArray(tm, "repo"),
  }
}

// ── Decoder ──

/** Validate the extracted object against the schema. Returns `null` on failure. */
const decodeMetadata = Schema.decodeUnknownOption(ExpandedGgufMetadataSchema)

// ── Cache ──

interface CachedMetadata {
  readonly size: number
  readonly mtimeMs: number
  readonly metadata: ExpandedGgufMetadata | null
}

const metadataCache = new Map<string, CachedMetadata>()

// ── Public API ──

/**
 * Read expanded GGUF metadata from a file.
 *
 * Uses `@huggingface/gguf` with `typedMetadata: true` and `computeParametersCount: true`
 * for precise, non-heuristic data extraction. The library's flat KV record is mapped
 * to the structured shape and validated via Effect Schema (`ExpandedGgufMetadataSchema`).
 *
 * Results are cached by (size, mtime). Corrupted files return `null`.
 */
export function readGgufMetadata(
  filePath: string,
): Effect.Effect<ExpandedGgufMetadata | null, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem

    const exists = yield* fs.exists(filePath).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return null

    const stat = yield* fs.stat(filePath).pipe(Effect.catchAll(() => Effect.succeed(null)))
    if (!stat || stat.type !== "File") return null

    // Check stat-based cache
    const size = Number(stat.size)
    const mtimeMs = Option.getOrUndefined(stat.mtime)?.getTime() ?? 0
    const cached = metadataCache.get(filePath)
    if (cached && cached.size === size && cached.mtimeMs === mtimeMs) {
      return cached.metadata
    }

    // Read via @huggingface/gguf (header only, ~1-20ms)
    const metadata = yield* Effect.tryPromise({
      try: async () => {
        const parsed = await gguf(filePath, {
          allowLocalFile: true,
          typedMetadata: true,
          computeParametersCount: true,
        })
        const tm = parsed.typedMetadata as TypedMetadata
        const structured = toStructuredMetadata(tm)
        const decoded = decodeMetadata(structured)
        if (decoded._tag !== "Some") return null
        // parameterCount comes from the library's computation, not from metadata keys
        return { ...decoded.value, parameterCount: parsed.parameterCount } as ExpandedGgufMetadata
      },
      catch: () => null as ExpandedGgufMetadata | null,
    }).pipe(Effect.catchAll(() => Effect.succeed(null)))

    metadataCache.set(filePath, { size, mtimeMs, metadata })
    return metadata
  })
}
