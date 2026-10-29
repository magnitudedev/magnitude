import { BunContext } from "@effect/platform-bun"
import { Effect, Either } from "effect"
import { chmod, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { renderNodeHint, requireWritableDataDirectory, unopenableRenderNodes } from "./container"

describe("container data directory", () => {
  it.skipIf(process.platform === "win32" || process.getuid?.() === 0)("names the fix when the mounted volume belongs to another user", async () => {
    const volume = await mkdtemp(join(tmpdir(), "magnitude-volume-"))
    const check = (directory: string) => Effect.runPromise(requireWritableDataDirectory(directory).pipe(Effect.either, Effect.provide(BunContext.layer)))
    try {
      expect(Either.isRight(await check(join(volume, ".magnitude")))).toBe(true)
      await rm(join(volume, ".magnitude"), { recursive: true })
      await chmod(volume, 0o555)
      const refused = await check(join(volume, ".magnitude"))
      expect(Either.isLeft(refused) && refused.left.message).toBe(`Magnitude can't write its data directory ${join(volume, ".magnitude")}. On the host, run \`chown -R 1000:1000\` on the directory mounted at `
        + `${volume} or start the container with \`--user <uid>\` of its owner; with SELinux, also add \`:Z\` to the volume.`)
    } finally {
      await chmod(volume, 0o755)
      await rm(volume, { recursive: true, force: true })
    }
  })
  it.skipIf(process.platform === "win32" || process.getuid?.() === 0)("names the render nodes the container's user can't open", async () => {
    const dri = await mkdtemp(join(tmpdir(), "magnitude-dri-"))
    const nodes = () => Effect.runPromise(unopenableRenderNodes(dri).pipe(Effect.provide(BunContext.layer)))
    try {
      for (const name of ["card0", "renderD128", "renderD129"]) await writeFile(join(dri, name), "")
      expect(await nodes()).toEqual([])
      await chmod(join(dri, "renderD129"), 0o000)
      await chmod(join(dri, "card0"), 0o000)
      expect(await nodes()).toEqual([join(dri, "renderD129")])
      expect(await Effect.runPromise(unopenableRenderNodes(join(dri, "absent")).pipe(Effect.provide(BunContext.layer)))).toEqual([])
      expect(renderNodeHint("/dev/dri/renderD128")).toContain("--group-add $(stat -c %g /dev/dri/renderD128)")
    } finally { await rm(dri, { recursive: true, force: true }) }
  })
})
