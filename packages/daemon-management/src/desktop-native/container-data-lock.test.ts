import { BunContext } from "@effect/platform-bun"
import { Effect, Either, Option } from "effect"
import { chmod, mkdtemp, rm, stat, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { holdContainerData } from "./container-data-lock"
import { NativeHost, NativeHostUnavailable } from "./index"

const native = (acquire: (path: string) => Effect.Effect<Option.Option<object>, NativeHostUnavailable>) => NativeHost.of({
  requireInteractiveDesktop: Effect.void, isServiceSession: Effect.succeed(false), localAppDataDirectory: Effect.succeed(""),
  acquireOwnership: acquire, guardParent: () => Effect.void, ownedEndpoint: () => Effect.succeed(""), inspectEndpoint: () => Effect.succeed(Option.none()),
})
const hold = (dataDirectory: string, host: NativeHost) => Effect.runPromise(holdContainerData(dataDirectory).pipe(
  Effect.either, Effect.scoped, Effect.provideService(NativeHost, host), Effect.provide(BunContext.layer)))

describe("container data lock", () => {
  it.skipIf(process.platform === "win32")("makes a lock file Kubernetes' fsGroup opened to its group private again before locking it", async () => {
    const data = await mkdtemp(join(tmpdir(), "magnitude-volume-"))
    try {
      await writeFile(join(data, "container.lock"), "")
      await chmod(join(data, "container.lock"), 0o660)
      const modes: number[] = []
      const result = await hold(data, native(path => Effect.promise(() => stat(path)).pipe(Effect.map(info => { modes.push(info.mode & 0o777); return Option.some({}) }))))
      expect(Either.isRight(result)).toBe(true)
      expect(modes).toEqual([0o600])
    } finally { await rm(data, { recursive: true, force: true }) }
  })
  it("refuses a volume another container holds, and a lock it can't safely use, but goes on where locking is unsupported", async () => {
    const data = await mkdtemp(join(tmpdir(), "magnitude-volume-"))
    try {
      const held = await hold(data, native(() => Effect.succeed(Option.none())))
      expect(Either.isLeft(held) && held.left._tag).toBe("ContainerDataInUse")
      const unsafe = await hold(data, native(() => Effect.fail(new NativeHostUnavailable({ message: "Unsafe ownership lock file" }))))
      expect(Either.isLeft(unsafe) && unsafe.left.message).toContain("which another user created")
      expect(Either.isRight(await hold(data, native(() => Effect.fail(new NativeHostUnavailable({ message: "Cannot acquire ownership lock" })))))).toBe(true)
    } finally { await rm(data, { recursive: true, force: true }) }
  })
})
