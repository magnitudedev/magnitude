import { stat } from "node:fs/promises"
import { Effect } from "effect"
import { gguf, type MetadataValue } from "@huggingface/gguf"

export interface LocalGgufMetadata {
  readonly generalName?: string
  readonly generalBasename?: string
  readonly generalSizeLabel?: string
  readonly generalFinetune?: string
  readonly generalVersion?: string
  readonly architecture?: string
  readonly tokenizerModel?: string
  readonly tokenizerPre?: string
  readonly baseModelNames: readonly string[]
  readonly baseModelRepositories: readonly string[]
}

interface CachedMetadata {
  readonly size: number
  readonly mtimeMs: number
  readonly metadata: LocalGgufMetadata | null
}

const cache = new Map<string, CachedMetadata>()
const GGUF_EXTENSION = /\.gguf$/i

function stringValue(value: MetadataValue | undefined): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined
}

function unique(values: readonly (string | undefined)[]): readonly string[] {
  return [...new Set(values.filter((value): value is string => value !== undefined))]
}

function selectMetadata(metadata: Record<string, MetadataValue>): LocalGgufMetadata {
  const baseModelNames: string[] = []
  const baseModelRepositories: string[] = []

  for (const [key, value] of Object.entries(metadata)) {
    if (/^general\.base_model\.\d+\.name$/i.test(key)) {
      baseModelNames.push(stringValue(value) ?? "")
    } else if (/^general\.base_model\.\d+\.(?:repo_url|url)$/i.test(key)) {
      baseModelRepositories.push(stringValue(value) ?? "")
    }
  }

  const generalName = stringValue(metadata["general.name"])
  const generalBasename = stringValue(metadata["general.basename"])
  const generalSizeLabel = stringValue(metadata["general.size_label"])
  const generalFinetune = stringValue(metadata["general.finetune"])
  const generalVersion = stringValue(metadata["general.version"])
  const architecture = stringValue(metadata["general.architecture"])
  const tokenizerModel = stringValue(metadata["tokenizer.ggml.model"])
  const tokenizerPre = stringValue(metadata["tokenizer.ggml.pre"])

  return {
    ...(generalName ? { generalName } : {}),
    ...(generalBasename ? { generalBasename } : {}),
    ...(generalSizeLabel ? { generalSizeLabel } : {}),
    ...(generalFinetune ? { generalFinetune } : {}),
    ...(generalVersion ? { generalVersion } : {}),
    ...(architecture ? { architecture } : {}),
    ...(tokenizerModel ? { tokenizerModel } : {}),
    ...(tokenizerPre ? { tokenizerPre } : {}),
    baseModelNames: unique(baseModelNames),
    baseModelRepositories: unique(baseModelRepositories),
  }
}

/**
 * Read selected metadata from a locally accessible GGUF file.
 * Missing files, container-only paths, and malformed files are normal fallbacks.
 */
export function readLocalGgufMetadata(
  modelPath: string | undefined,
): Effect.Effect<LocalGgufMetadata | null> {
  if (!modelPath || !GGUF_EXTENSION.test(modelPath.trim())) {
    return Effect.succeed(null)
  }

  const path = modelPath.trim()
  return Effect.tryPromise(async () => {
    const file = await stat(path)
    if (!file.isFile()) return null

    const cached = cache.get(path)
    if (cached && cached.size === file.size && cached.mtimeMs === file.mtimeMs) {
      return cached.metadata
    }

    const parsed = await gguf(path, { allowLocalFile: true })
    const metadata = selectMetadata(parsed.metadata as Record<string, MetadataValue>)
    cache.set(path, { size: file.size, mtimeMs: file.mtimeMs, metadata })
    return metadata
  }).pipe(Effect.catchAll(() => Effect.succeed(null)))
}
