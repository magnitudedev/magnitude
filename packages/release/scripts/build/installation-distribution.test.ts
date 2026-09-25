import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { UpdateManifest, signUpdateManifest } from "../../src/hosted-update/manifest"
import { InstallationOffer, verifyInstallationOffer } from "../../src/hosted-update/installation-offer"
import { writeInstallationDistribution } from "./installation-distribution"

describe("static installation distribution", () => {
  it.each(["valid", "tampered", "duplicate"])("prepares only a verified %s batch", scenario => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const root = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-install-distribution-" })
    const output = join(root, "hosting")
    const keys = yield* Effect.sync(() => generateKeyPairSync("ed25519"))
    const target = { os: "linux", arch: "arm64", package: "deb" } as const
    const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version: "0.1.6", tag: "@magnitudedev/cli@0.1.6", commit: "a".repeat(40),
      artifact: { id: "desktop-linux-arm64-deb", target, filename: "magnitude.deb", bytes: 12, sha256: "a".repeat(64) } })
    const publication = yield* signUpdateManifest(manifest, keys.privateKey)
    const candidate = scenario === "tampered" ? { ...publication, release: { ...publication.release, bytes: 13 } } : publication
    const options = { output, origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ", windowsPublisher: "Magnitude",
      publicKey: keys.publicKey.export({ type: "spki", format: "pem" }).toString(), publications: scenario === "duplicate" ? [candidate, candidate] : [candidate] }
    if (scenario !== "valid") {
      expect(yield* writeInstallationDistribution(options).pipe(Effect.isFailure)).toBe(true)
      expect(yield* fs.exists(output)).toBe(false)
      return
    }
    expect(yield* writeInstallationDistribution(options)).toEqual({ version: "0.1.6", channel: "stable", offers: 1 })
    const offer = yield* Schema.decodeUnknown(Schema.parseJson(InstallationOffer))(yield* fs.readFileString(join(output, "install/stable/linux-arm64-deb.json")))
    yield* verifyInstallationOffer(offer, target, "stable", new Map([["publisher", keys.publicKey]]))
    expect(yield* fs.exists(join(output, "install.sh"))).toBe(true)
    expect(yield* fs.exists(join(output, "install.ps1"))).toBe(true)
    expect(yield* writeInstallationDistribution(options).pipe(Effect.isFailure)).toBe(true)
  })).pipe(Effect.provide(NodeContext.layer))))
})
