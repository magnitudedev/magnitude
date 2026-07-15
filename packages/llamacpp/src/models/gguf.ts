import { Data, Effect, Option, Schema } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { gguf } from "@huggingface/gguf"
import { GGMLFileQuantizationType } from "@huggingface/tasks"
import {
  ExpandedGgufMetadata as ExpandedGgufMetadataSchema,
  type ExpandedGgufMetadata,
} from "./types"

// ── Field extraction from the library's typed metadata ──

type TypedMetadata = Record<string, { value: unknown; type: number; subType?: number }>

const isRecord = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null

const normalizeTypedMetadata = (value: unknown): TypedMetadata => {
  if (!isRecord(value)) return {}
  const metadata: TypedMetadata = {}
  for (const [key, candidate] of Object.entries(value)) {
    if (!isRecord(candidate) || !("value" in candidate) || typeof candidate.type !== "number") continue
    metadata[key] = {
      value: candidate.value,
      type: candidate.type,
      ...(typeof candidate.subType === "number" ? { subType: candidate.subType } : {}),
    }
  }
  return metadata
}

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
  const quantization = fileType === undefined
    ? undefined
    : Object.entries(GGMLFileQuantizationType).find(
      ([key, value]) => Number(key) === fileType && typeof value === "string",
    )?.[1]

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

class GgufReadError extends Data.TaggedError("GgufReadError")<{
  readonly cause: unknown
}> {}

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
      try: async (): Promise<ExpandedGgufMetadata | null> => {
        const parsed = await gguf(filePath, {
          allowLocalFile: true,
          typedMetadata: true,
          computeParametersCount: true,
        })
        const tm = normalizeTypedMetadata(parsed.typedMetadata)
        const structured = toStructuredMetadata(tm)
        const decoded = decodeMetadata(structured)
        if (decoded._tag !== "Some") return null
        // parameterCount comes from the library's computation, not from metadata keys
        const parameterCount = typeof parsed.parameterCount === "bigint"
          ? Number(parsed.parameterCount)
          : parsed.parameterCount
        return {
          ...decoded.value,
          ...(typeof parameterCount === "number" && Number.isFinite(parameterCount)
            ? { parameterCount }
            : {}),
        }
      },
      catch: (cause) => new GgufReadError({ cause }),
    }).pipe(Effect.catchAll(() => Effect.succeed(null)))

    metadataCache.set(filePath, { size, mtimeMs, metadata })
    return metadata
  })
}
