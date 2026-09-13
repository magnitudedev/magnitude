import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { PublisherKeyId, signUpdateManifest, UpdateManifest } from "../../packages/release/src/hosted-update/manifest"
import { UpdateClientMetadata } from "@magnitudedev/release/hosted-update"
import { LinuxPackageUpdate } from "@magnitudedev/daemon-management/desktop-native"
import { makeLinuxUpdateSource } from "./linux-update-source"

describe("Linux update staging", () => {
  it("retains original publisher proof and package bytes, then discards them on ordinary Quit", async () => {
    const directory = await mkdtemp(join(tmpdir(), "linux-update-stage-"))
    const archive = join(directory, "download.deb")
    await writeFile(archive, "verified-by-shared-download")
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "linux", target: { os: "linux", arch: "arm64", package: "deb" }, path: "releases/2.0.0/desktop.deb", bytes: 26, sha256: "a".repeat(64),
    } })
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, PublisherKeyId.make("test"), generateKeyPairSync("ed25519").privateKey))
    try {
      await Effect.runPromise(Effect.gen(function* () {
        const linux = yield* makeLinuxUpdateSource({ origin: "https://magnitude.dev", storageOrigin: "https://storage.example", trustedPublishers: new Map(),
          metadata: yield* Schema.decodeUnknown(UpdateClientMetadata)({ version: "1.0.0", os: "linux", os_version: "6.1", arch: "arm64", package: "deb" }),
          sign: () => Effect.succeed("unused"), userAgent: "fixture", cacheDirectory: join(directory, "cache"), stateDirectory: directory })
        expect(linux.previousFailure._tag).toBe("None")
        expect((yield* linux.restart(false).pipe(Effect.either))._tag).toBe("Left")
        yield* linux.source.stage(archive, { manifest, envelope })
        yield* Effect.promise(async () => {
          const entries = await readdir(join(directory, "application-updates"))
          expect(entries).toHaveLength(1)
          const request = Schema.decodeUnknownSync(Schema.parseJson(LinuxPackageUpdate))(await readFile(join(directory, "application-updates", entries[0]!, "request.json"), "utf8"))
          expect(request.envelope).toEqual(envelope)
          expect(await readFile(request.packagePath, "utf8")).toBe("verified-by-shared-download")
        })
        yield* linux.discard
        expect(yield* Effect.promise(() => readdir(join(directory, "application-updates")))).toEqual([])
      }).pipe(Effect.provide(NodeContext.layer)))
    } finally { await rm(directory, { recursive: true, force: true }) }
  })
})
