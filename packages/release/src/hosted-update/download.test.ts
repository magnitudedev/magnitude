import { generateKeyPairSync } from "node:crypto"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { handleArtifactDownload } from "./download"
import { DistributionStore, DistributionStoreUnavailable } from "./service"
import { PublisherKeyId, signUpdateManifest, UpdateManifest } from "./manifest"
import { newUpdateNonce, signUpdateRequest, updateQuery } from "./request-auth"

const keys = generateKeyPairSync("ed25519"), publisher = generateKeyPairSync("ed25519"), keyId = PublisherKeyId.make("test")
const options = { origin: "https://magnitude.dev", storageOrigin: "https://downloads.magnitude.dev", country: Option.some("US"), trustedPublishers: new Map([[keyId, publisher.publicKey]]) }
const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: { id: "mac", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, path: "releases/2.0.0/mac.zip", bytes: 42, sha256: "a".repeat(64) } })
const request = async (fields: Record<string, string> = {}) => {
  const url = new URL(options.origin + "/api/download/mac?" + updateQuery({ protocol: "1", product: "desktop", version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip", channel: "stable", ts: String(Math.floor(Date.now() / 1000)), nonce: await Effect.runPromise(newUpdateNonce), release: "2.0.0", ...fields }))
  return new Request(url, { headers: { authorization: await Effect.runPromise(signUpdateRequest(keys.privateKey, url)) } })
}
const harness = async (overrides: Partial<DistributionStore> = {}) => {
  const envelope = await Effect.runPromise(signUpdateManifest(manifest, keyId, publisher.privateKey))
  let recorded = 0
  const seen = new Set<string>()
  const store: DistributionStore = {
    admit: (_, nonce) => Effect.sync(() => { if (seen.has(nonce)) return false; seen.add(nonce); return true }),
    artifact: (version, id) => Effect.succeed(version === manifest.version && id === manifest.artifact.id ? Option.some(envelope) : Option.none()),
    recordDownload: () => Effect.sync(() => { recorded++ }),
    candidates: () => Effect.succeed([]), recordCheck: () => Effect.void, ...overrides,
  }
  return { count: () => recorded, run: (req: Request) => Effect.runPromise(handleArtifactDownload(req, options).pipe(Effect.provideService(DistributionStore, store))) }
}
describe("authenticated artifact downloads", () => {
  it("records one intent and redirects to the signed immutable object without credentials", async () => {
    const h = await harness(), req = await request(), result = await h.run(req)
    expect(result.status).toBe(302)
    expect(result.headers.get("location")).toBe("https://downloads.magnitude.dev/releases/2.0.0/mac.zip")
    expect(result.headers.get("authorization")).toBeNull()
    expect(result.headers.get("cache-control")).toBe("private, no-store")
    expect((await h.run(req)).status).toBe(409)
    expect(h.count()).toBe(1)
  })
  it("rejects wrong releases, targets, signatures and unsigned requests before recording", async () => {
    const h = await harness()
    expect((await h.run(await request({ release: "3.0.0" }))).status).toBe(404)
    expect((await h.run(await request({ arch: "x64" }))).status).toBe(404)
    const signed = await request(), url = new URL(signed.url)
    url.searchParams.set("release", "3.0.0")
    expect((await h.run(new Request(url, { headers: signed.headers }))).status).toBe(401)
    expect((await h.run(new Request(signed.url))).status).toBe(401)
    expect(h.count()).toBe(0)
  })
  it("still offers the verified object when telemetry writing fails", async () => {
    const h = await harness({ recordDownload: () => new DistributionStoreUnavailable() })
    expect((await h.run(await request())).status).toBe(302)
  })
})
