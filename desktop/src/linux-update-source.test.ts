import { NodeContext } from "@effect/platform-node"
import { Effect, Layer, Option, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../packages/release/src/hosted-update/release"
import { UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { makePreparedUpdateStore, PreparedUpdateStore, unixPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { makeLinuxUpdateSource } from "./linux-update-source"
import { installPreparedUpdate, PreparedUpdateInstaller } from "./prepared-update-installation"

describe("Linux prepared update", () => {
  it("retains one signed package across owner exit and defers background authorization", async () => {
    const directory = await mkdtemp(join(tmpdir(), "linux-update-stage-"))
    const archive = join(directory, "download.deb")
    const bytes = Buffer.from("verified-by-shared-download")
    await writeFile(archive, bytes)
    const publisher = generateKeyPairSync("ed25519")
    const target = { os: "linux", arch: "arm64", package: "deb" } as const
    const release = await Effect.runPromise(signUpdateRelease({ version: "2.0.0", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex") }, target, publisher.privateKey))
    const options = { dataDirectory: directory, target, trustedPublishers: new Map([["test", publisher.publicKey]]) }
    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const store = yield* makePreparedUpdateStore(options)
        const linux = yield* makeLinuxUpdateSource({ origin: "https://magnitude.dev", trustedPublishers: options.trustedPublishers,
          metadata: yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: "1.0.0", os: "linux", os_version: "6.1", arch: "arm64", package: "deb" }),
          sign: () => Effect.succeed("unused"), userAgent: "fixture", cacheDirectory: join(directory, "updates"), dataDirectory: directory,
          stateDirectory: join(directory, "state") }).pipe(Effect.provideService(PreparedUpdateStore, store))
        yield* linux.source.stage(archive, release)
        expect(yield* installPreparedUpdate({ showWindow: false, allowAuthorizationPrompt: false }).pipe(Effect.provideService(PreparedUpdateStore, store),
          Effect.provideService(PreparedUpdateInstaller, linux.installer))).toBe("Deferred")
        expect(Option.getOrThrow(yield* store.read).installation._tag).toBe("Unattempted")
      })).pipe(Effect.provide(unixPrivateFilePermissions.pipe(Layer.provideMerge(NodeContext.layer)))))
      expect((await readdir(join(directory, "updates"))).sort()).toEqual(["magnitude.deb", "update.json"])
      expect(await readFile(join(directory, "updates", "magnitude.deb"))).toEqual(bytes)
      const pending = await Effect.runPromise(makePreparedUpdateStore(options).pipe(Effect.flatMap(store => store.read),
        Effect.provide(unixPrivateFilePermissions.pipe(Layer.provideMerge(NodeContext.layer)))))
      expect(Option.getOrThrow(pending)).toEqual({ release, installation: { _tag: "Unattempted" } })
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
