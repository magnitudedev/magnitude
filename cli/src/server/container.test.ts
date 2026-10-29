import { BunContext } from "@effect/platform-bun"
import { Effect, Either } from "effect"
import { chmod, mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { requireWritableDataDirectory } from "./container"

describe("container data directory", () => {
  it.skipIf(process.platform === "win32" || process.getuid?.() === 0)("names the fix when the mounted volume belongs to another user", async () => {
    const volume = await mkdtemp(join(tmpdir(), "magnitude-volume-"))
    const check = (directory: string) => Effect.runPromise(requireWritableDataDirectory(directory).pipe(Effect.either, Effect.provide(BunContext.layer)))
    try {
      expect(Either.isRight(await check(join(volume, ".magnitude")))).toBe(true)
      await rm(join(volume, ".magnitude"), { recursive: true })
      await chmod(volume, 0o555)
      const refused = await check(join(volume, ".magnitude"))
      expect(Either.isLeft(refused) && refused.left.message).toBe(`Magnitude can't write its data directory ${join(volume, ".magnitude")}, which belongs to another user. `
        + `On the host, run \`chown -R 1000:1000\` on the directory mounted at ${volume}, or start the container with \`--user <uid>\` of its owner.`)
    } finally {
      await chmod(volume, 0o755)
      await rm(volume, { recursive: true, force: true })
    }
  })
})
