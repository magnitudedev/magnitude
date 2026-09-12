import { generateKeyPairSync } from "node:crypto"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { handleInstallerDownload } from "./installer"
import { PublisherKeyId, signUpdateManifest, UpdateManifest, type SignedUpdateManifest } from "./manifest"
import { DistributionStore, DistributionStoreUnavailable, type InstallerDownloadObservation } from "./service"

const publisher = generateKeyPairSync("ed25519"), keyId = PublisherKeyId.make("test")
const options = { origin: "https://magnitude.dev", storageOrigin: "https://downloads.magnitude.dev", country: Option.some("US"), trustedPublishers: new Map([[keyId, publisher.publicKey]]) }
const signed = (version = "2.0.0", pkg = "dmg") => Effect.runPromise(signUpdateManifest(Schema.decodeUnknownSync(UpdateManifest)({
  protocol: 1, version, commit: "a".repeat(40), artifact: { id: "mac", target: { os: "darwin", arch: "arm64", package: pkg }, path: `releases/${version}/mac.${pkg}`, bytes: 100, sha256: "a".repeat(64) },
}), keyId, publisher.privateKey))
const harness = (candidates: readonly SignedUpdateManifest[], overrides: Partial<DistributionStore> = {}) => {
  const records: (typeof InstallerDownloadObservation.Type)[] = []
  const store: DistributionStore = {
    admit: () => Effect.die("Public downloads must not create installation identities"),
    recordCheck: () => Effect.die("Public downloads are not update checks"),
    recordDownload: () => Effect.die("Public downloads have no authenticated installation"),
    artifact: () => Effect.succeed(Option.none()), candidates: () => Effect.succeed(candidates),
    recordInstallerDownload: value => Effect.sync(() => { records.push(value) }), ...overrides,
  }
  return { records, run: (query = "os=darwin&arch=arm64&package=dmg", method = "GET") => Effect.runPromise(handleInstallerDownload(new Request(`${options.origin}/api/installer?${query}`, { method }), options).pipe(Effect.provideService(DistributionStore, store))) }
}
describe("public installers", () => {
  it("selects the newest stable installer and records anonymous download intent", async () => {
    const h = harness([await signed("1.0.0"), await signed("3.0.0-beta.1"), await signed(), await signed("4.0.0", "mac-zip")])
    const response = await h.run()
    expect(response.status).toBe(302)
    expect(response.headers.get("location")).toBe("https://downloads.magnitude.dev/releases/2.0.0/mac.dmg")
    expect(response.headers.get("cache-control")).toBe("private, no-store")
    expect(h.records).toHaveLength(1)
    expect(Object.keys(h.records[0]!)).toEqual(["target", "release", "artifact", "country"])
    expect(h.records[0]?.country).toEqual(Option.some("US"))
  })
  it("rejects duplicate, unknown, incompatible and update-only targets", async () => {
    const h = harness([await signed()])
    for (const query of ["os=darwin&arch=arm64&package=dmg&arch=x64", "os=darwin&arch=arm64&package=dmg&extra=x", "os=windows&arch=arm64&package=windows-exe", "os=darwin&arch=arm64&package=mac-zip"]) expect((await h.run(query)).status).toBe(400)
    expect((await h.run(undefined, "POST")).status).toBe(405)
    expect(h.records).toHaveLength(0)
  })
  it("does not serve absent, mismatched or untrusted artifacts", async () => {
    expect((await harness([]).run()).status).toBe(404)
    expect((await harness([await signed()]).run("os=darwin&arch=x64&package=dmg")).status).toBe(404)
    const envelope = await signed()
    expect((await harness([{ ...envelope, signature: "invalid" }]).run()).status).toBe(503)
  })
  it("serves verified installers through telemetry failure but fails closed on release-store failure", async () => {
    expect((await harness([await signed()], { recordInstallerDownload: () => new DistributionStoreUnavailable() }).run()).status).toBe(302)
    expect((await harness([], { candidates: () => new DistributionStoreUnavailable() }).run()).status).toBe(503)
  })
})
