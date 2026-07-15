import { basename } from "node:path"
import { parseGgufShardFilename } from "@huggingface/gguf"
import type { ShardGroup } from "./types"

/**
 * Group a list of file paths into multi-shard groups and singletons.
 * Uses `@huggingface/gguf`'s `parseGgufShardFilename` for shard filename parsing.
 */
export function groupShards(files: readonly string[]): {
  readonly groups: readonly ShardGroup[]
  readonly singletons: readonly string[]
} {
  const groupMap = new Map<string, string[]>()
  const singletons: string[] = []

  for (const file of files) {
    const name = basename(file)
    const parsed = parseGgufShardFilename(name)
    if (parsed) {
      const key = `${parsed.prefix}|${parsed.total}`
      const arr = groupMap.get(key) ?? []
      arr.push(file)
      groupMap.set(key, arr)
    } else {
      singletons.push(file)
    }
  }

  const groups: ShardGroup[] = Array.from(groupMap.entries()).map(([key, shards]) => {
    const [prefix, total] = key.split("|")
    const sorted = [...shards].sort((a, b) => basename(a).localeCompare(basename(b)))
    return {
      prefix,
      total: parseInt(total, 10),
      shards: sorted,
      primaryPath: sorted[0],
    }
  })

  return { groups, singletons }
}
