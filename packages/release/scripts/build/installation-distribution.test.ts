import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { UpdateManifest, signUpdateManifest } from "../../src/hosted-update/manifest"
import { InstallationOffer, verifyInstallationOffer } from "../../src/hosted-update/installation-offer"
import { writeInstallationDistribution } from "./installation-distribution"
import { writeInstallerOfferFixture } from "../acceptance/installer-offer-fixture"

const keys = generateKeyPairSync("ed25519")
const publicKey = keys.publicKey.export({ type: "spki", format: "pem" }).toString()
const options = (output: string) => ({ output, origin: "https://magnitude.dev", appleTeam: "ABCDEFGHIJ", windowsPublisher: "Magnitude", publicKey })

describe("installation distribution", () => {
  it("prepares only the two scripts, which fetch counted offers, and never overwrites a hosting tree", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const output = join(yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-install-distribution-" }), "hosting")
    yield* writeInstallationDistribution(options(output))
    expect((yield* fs.readDirectory(output)).sort()).toEqual(["install.ps1", "install.sh"])
    expect(yield* fs.readFileString(join(output, "install.sh"))).toContain("$origin/api/installer?os=$1&arch=$arch&package=$2&offer=1")
    expect(yield* fs.readFileString(join(output, "install.ps1"))).toContain("$origin/api/installer?os=windows&arch=x64&package=windows-exe&offer=1")
    expect(yield* writeInstallationDistribution(options(output)).pipe(Effect.isFailure)).toBe(true)
  })).pipe(Effect.provide(NodeContext.layer))))
  it("rejects an invalid publisher key before writing", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const output = join(yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-install-distribution-" }), "hosting")
    expect(yield* writeInstallationDistribution({ ...options(output), publicKey: "not a key" }).pipe(Effect.isFailure)).toBe(true)
    expect(yield* fs.exists(output)).toBe(false)
  })).pipe(Effect.provide(NodeContext.layer))))
  it.each(["valid", "tampered"])("serves only a verified %s offer from the acceptance fixture endpoint", scenario => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const hosting = yield* fs.makeTempDirectoryScoped({ prefix: "magnitude-offer-fixture-" })
    const target = { os: "linux", arch: "arm64", package: "deb" } as const
    const manifest = yield* Schema.decodeUnknown(UpdateManifest)({ protocol: 1, version: "0.1.6", tag: "@magnitudedev/cli@0.1.6", commit: "a".repeat(40),
      artifact: { id: "desktop-linux-arm64-deb", target, filename: "magnitude.deb", bytes: 12, sha256: "a".repeat(64) } })
    const publication = yield* signUpdateManifest(manifest, keys.privateKey)
    const written = yield* writeInstallerOfferFixture(hosting, scenario === "tampered" ? { ...publication, release: { ...publication.release, bytes: 13 } } : publication, publicKey).pipe(Effect.either)
    if (scenario === "tampered") return expect(written._tag === "Left" && !(yield* fs.exists(join(hosting, "api/installer")))).toBe(true)
    const offer = yield* Schema.decodeUnknown(Schema.parseJson(InstallationOffer))(yield* fs.readFileString(join(hosting, "api/installer")))
    yield* verifyInstallationOffer(offer, target, "stable", new Map([["publisher", keys.publicKey]]))
  })).pipe(Effect.provide(NodeContext.layer))))
})
