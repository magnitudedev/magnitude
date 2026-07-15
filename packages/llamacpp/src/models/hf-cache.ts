import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { homedir } from "node:os"
import { join } from "node:path"
import type { LocalModelInfo } from "./types"

const HF_REPO_FOLDER_RE = /^models--(.+)$/

/**
 * Parse a HF Hub cache folder name into a repo ID.
 * `models--unsloth--gemma-3-4b-it-GGUF` → `unsloth/gemma-3-4b-it-GGUF`
 */
export function parseRepoFolder(folder: string): string | null {
  const match = folder.match(HF_REPO_FOLDER_RE)
  if (!match) return null
  return match[1].replace(/--/g, "/")
}

/**
 * Resolve the HuggingFace Hub cache directory.
 * Checks env vars in order of precedence, then falls back to the default.
 */
export function hfCacheDir(): string {
  const env = process.env
  if (env.HF_HUB_CACHE) return env.HF_HUB_CACHE
  if (env.HUGGINGFACE_HUB_CACHE) return env.HUGGINGFACE_HUB_CACHE
  if (env.HF_HOME) return join(env.HF_HOME, "hub")
  if (env.XDG_CACHE_HOME) return join(env.XDG_CACHE_HOME, "huggingface", "hub")
  return join(homedir(), ".cache", "huggingface", "hub")
}

/**
 * Scan a HuggingFace Hub cache directory for downloaded GGUF models.
 * Returns `LocalModelInfo` entries with `source: { type: "hf-cache", ... }`.
 */
export function scanHfCache(
  cacheDir: string,
  scanDir: (dir: string) => Effect.Effect<readonly LocalModelInfo[], never, FileSystem.FileSystem | Path.Path>,
): Effect.Effect<readonly LocalModelInfo[], never, FileSystem.FileSystem | Path.Path> {
  return Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const pathSvc = yield* Path.Path

    const exists = yield* fs.exists(cacheDir).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (!exists) return []

    const entries = yield* fs.readDirectory(cacheDir).pipe(
      Effect.catchAll(() => Effect.succeed([] as readonly string[])),
    )

    const results: LocalModelInfo[] = []

    for (const entry of entries) {
      const repoId = parseRepoFolder(entry)
      if (!repoId) continue

      const repoDir = pathSvc.join(cacheDir, entry)
      const snapshotsDir = pathSvc.join(repoDir, "snapshots")
      const snapshotsExist = yield* fs.exists(snapshotsDir).pipe(
        Effect.catchAll(() => Effect.succeed(false)),
      )
      if (!snapshotsExist) continue

      const commits = yield* fs.readDirectory(snapshotsDir).pipe(
        Effect.catchAll(() => Effect.succeed([] as readonly string[])),
      )

      for (const commit of commits) {
        const snapshotDir = pathSvc.join(snapshotsDir, commit)
        const models = yield* scanDir(snapshotDir)
        for (const model of models) {
          results.push({
            ...model,
            source: { _tag: "hf-cache" as const, repoId, commit },
            repoId,
            commit,
          })
        }
      }
    }

    return results
  })
}
