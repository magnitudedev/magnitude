import { NodeContext } from "@effect/platform-node"
import { Effect, Either, Layer } from "effect"
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { makeUpdateIdentity } from "./update-identity"
import { unixPrivateFilePermissions } from "@magnitudedev/daemon-management/private-files"

describe("desktop installation identity", () => {
  it("persists one private identity across launches and refuses to reset a corrupt key", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-update-identity-"))
    const identity = makeUpdateIdentity(root).pipe(Effect.provide(unixPrivateFilePermissions.pipe(Layer.provideMerge(NodeContext.layer))))
    const make = () => Effect.runPromise(identity)
    const url = new URL("https://magnitude.dev/api/update?ts=1&nonce=test")
    try {
      const first = await make(), second = await make()
      expect(await Effect.runPromise(first.sign(url))).toBe(await Effect.runPromise(second.sign(url)))
      const path = join(root, "updates", "installation-key.pem")
      if (process.platform !== "win32") expect((await stat(path)).mode & 0o777).toBe(0o600)
      await writeFile(path, "corrupt")
      expect(Either.isLeft(await Effect.runPromise(identity.pipe(Effect.either)))).toBe(true)
      expect(await readFile(path, "utf8")).toBe("corrupt")
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
