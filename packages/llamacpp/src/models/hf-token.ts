import { Effect } from "effect"
import * as FileSystem from "@effect/platform/FileSystem"
import { homedir } from "node:os"
import { join } from "node:path"

/**
 * Resolve the HuggingFace access token.
 *
 * Priority:
 * 1. `HF_TOKEN` env var
 * 2. `~/.cache/huggingface/token` file
 * 3. Explicitly stored token from MagnitudeConfig
 */
export function resolveHfToken(
  storedToken?: string,
): Effect.Effect<string | null, never, FileSystem.FileSystem> {
  return Effect.gen(function* () {
    // 1. Env var
    if (process.env.HF_TOKEN?.trim()) return process.env.HF_TOKEN.trim()

    // 2. Token file
    const fs = yield* FileSystem.FileSystem
    const tokenFile = join(homedir(), ".cache", "huggingface", "token")
    const exists = yield* fs.exists(tokenFile).pipe(Effect.catchAll(() => Effect.succeed(false)))
    if (exists) {
      const content = yield* fs.readFileString(tokenFile).pipe(
        Effect.catchAll(() => Effect.succeed("")),
      )
      if (content.trim()) return content.trim()
    }

    // 3. Stored config token
    return storedToken?.trim() || null
  })
}
