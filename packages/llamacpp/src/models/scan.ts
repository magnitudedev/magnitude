import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { basename, dirname } from "node:path"
import type { LocalModelInfo, LocalModelSource, ShardGroup } from "./types"
import { groupShards } from "./shard"
import { pairMmproj } from "./mmproj"
import { readGgufMetadata } from "./gguf"

const GGUF_GLOB = /\.gguf$/i

/**
 * Scan a directory recursively for GGUF model files.
 * Groups shards, pairs mmproj projectors, reads metadata, and deduplicates.
 */
export function scanDirectory(
  dir: string,
  source: LocalModelSource,
): Effect.Effect<readonly LocalModelInfo[], never, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const pathSvc = yield* Path.Path

    const exists = yield* fs.exists(dir).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return []

    const allGgufFiles = yield* findGgufFiles(fs, pathSvc, dir)
    if (allGgufFiles.length === 0) return []

    // Group shards
    const { groups, singletons } = groupShards(allGgufFiles)

    // Pair mmproj files
    const mmprojMap = pairMmproj(allGgufFiles)

    const models: LocalModelInfo[] = []

    // Process shard groups
    for (const group of groups) {
      const model = yield* buildModelInfo(fs, pathSvc, group, source, mmprojMap)
      if (model) models.push(model)
    }

    // Process singletons (skip mmproj-only files)
    for (const file of singletons) {
      if (/mmproj/i.test(basename(file))) continue
      const singletonGroup: ShardGroup = {
        prefix: basename(file).replace(/\.gguf$/i, ""),
        total: 1,
        shards: [file],
        primaryPath: file,
      }
      const model = yield* buildModelInfo(fs, pathSvc, singletonGroup, source, mmprojMap)
      if (model) models.push(model)
    }

    return models
  })
}

/**
 * Recursively find all .gguf files under a directory (non-recursive into hidden dirs).
 */
function findGgufFiles(
  fs: FileSystem.FileSystem,
  pathSvc: Path.Path,
  dir: string,
): Effect.Effect<readonly string[], never, never> {
  return Effect.gen(function* () {
    const entries = yield* fs.readDirectory(dir).pipe(
      Effect.catchAll(() => Effect.succeed([] as readonly string[])),
    )

    const results: string[] = []

    for (const entry of entries) {
      // Skip hidden directories/files
      if (entry.startsWith(".")) continue

      const fullPath = pathSvc.join(dir, entry)
      const stat = yield* fs.stat(fullPath).pipe(Effect.catchAll(() => Effect.succeed(null)))
      if (!stat) continue

      if (stat.type === "Directory") {
        const nested = yield* findGgufFiles(fs, pathSvc, fullPath)
        results.push(...nested)
      } else if (stat.type === "File" && GGUF_GLOB.test(entry)) {
        results.push(fullPath)
      }
    }

    return results
  })
}

/**
 * Build a `LocalModelInfo` from a shard group (or singleton).
 * Reads metadata from the primary shard and computes total file size.
 */
function buildModelInfo(
  fs: FileSystem.FileSystem,
  pathSvc: Path.Path,
  group: ShardGroup,
  source: LocalModelSource,
  mmprojMap: Map<string, string>,
): Effect.Effect<LocalModelInfo | null, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    const primaryPath = group.primaryPath
    const stat = yield* fs.stat(primaryPath).pipe(Effect.catchAll(() => Effect.succeed(null)))
    if (!stat || stat.type !== "File") return null

    // Read metadata from primary shard
    const meta = yield* readGgufMetadata(primaryPath)

    // Compute total file size across all shards
    let fileSizeBytes = Number(stat.size)
    if (group.shards.length > 1) {
      for (const shard of group.shards.slice(1)) {
        const shardStat = yield* fs.stat(shard).pipe(Effect.catchAll(() => Effect.succeed(null)))
        if (shardStat && shardStat.type === "File") {
          fileSizeBytes += Number(shardStat.size)
        }
      }
    }

    const fileName = basename(primaryPath)
    const mmprojPath = mmprojMap.get(primaryPath)

    // Build model ID
    const id = buildModelId(primaryPath, source)

    // Display name: prefer metadata name, fall back to filename
    const displayName = meta?.generalName
      ?? meta?.generalBasename
      ?? fileName.replace(/\.gguf$/i, "")

    const hasMmproj = mmprojPath !== undefined
    const expertCount = meta?.expertCount

    return {
      id,
      displayName,
      filePath: primaryPath,
      shardPaths: group.shards.length > 1 ? group.shards : undefined,
      mmprojPath,
      architecture: meta?.architecture,
      quantization: meta?.quantization,
      contextLength: meta?.contextLength,
      fileSizeBytes,
      parameterCount: meta?.parameterCount,
      hiddenSize: meta?.hiddenSize,
      layerCount: meta?.layerCount,
      headCount: meta?.headCount,
      vocabSize: meta?.vocabSize,
      tokenizerModel: meta?.tokenizerModel,
      tokenizerPre: meta?.tokenizerPre,
      chatTemplate: meta?.chatTemplate,
      chatTemplatePresent: meta?.chatTemplatePresent ?? false,
      vision: hasMmproj || meta?.architecture === "mmproj",
      audio: false,
      moe: expertCount !== undefined && expertCount > 0,
      source,
      repoId: source._tag === "hf-cache" ? source.repoId : undefined,
      commit: source._tag === "hf-cache" ? source.commit : undefined,
      baseModelNames: meta?.baseModelNames.length ? meta.baseModelNames : undefined,
    }
  })
}

/**
 * Build a unique model ID from path and source.
 * HF cache: `repoId:filename`
 * User dir: `dir/filename`
 */
function buildModelId(filePath: string, source: LocalModelSource): string {
  const fileName = basename(filePath)
  if (source._tag === "hf-cache") {
    return `${source.repoId}:${fileName}`
  }
  const dir = dirname(filePath)
  return `${dir}/${fileName}`
}


