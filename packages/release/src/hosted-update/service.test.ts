import { generateKeyPairSync } from "node:crypto"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { DistributionStore, DistributionStoreUnavailable, handleUpdateCheck, type CheckObservation } from "./service"
import { newUpdateNonce, signUpdateRequest, updateQuery } from "./request-auth"
import { PublisherKeyId, UpdateManifest, signUpdateManifest, type SignedUpdateManifest } from "./manifest"

const installation = generateKeyPairSync("ed25519"), publisher = generateKeyPairSync("ed25519"), keyId = PublisherKeyId.make("publisher")
const options = { origin: "https://magnitude.dev", country: Option.some("US"), trustedPublishers: new Map([[keyId, publisher.publicKey]]) }
const request = async () => {
  const url = new URL(options.origin + "/api/update?" + updateQuery({ protocol: "1", product: "desktop", version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip", channel: "stable", ts: String(Math.floor(Date.now() / 1000)), nonce: await Effect.runPromise(newUpdateNonce) }))
  return new Request(url, { headers: { Authorization: await Effect.runPromise(signUpdateRequest(installation.privateKey, url)) } })
}
const signed = (version: string) => Effect.runPromise(signUpdateManifest(Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version, commit: "a".repeat(40), artifact: { id: "mac", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, path: `releases/${version}/mac.zip`, bytes: 100, sha256: "a".repeat(64) } }), keyId, publisher.privateKey))
const harness = (candidates: readonly SignedUpdateManifest[] = [], overrides: Partial<DistributionStore> = {}) => {
  const nonces = new Set<string>(), records: (typeof CheckObservation.Type)[] = []
  const store: DistributionStore = {
    admit: (id, nonce) => Effect.sync(() => { const key = `${id}:${nonce}`; if (nonces.has(key)) return false; nonces.add(key); return true }),
    candidates: () => Effect.succeed(candidates), recordCheck: value => Effect.sync(() => { records.push(value) }),
    artifact: () => Effect.succeed(Option.none()), recordDownload: () => Effect.void, ...overrides,
  }
  return { records, store, run: (req: Request) => Effect.runPromise(handleUpdateCheck(req, options).pipe(Effect.provideService(DistributionStore, store))) }
}
describe("hosted update checks", () => {
  it("records a verified current installation and returns uncached 204", async () => {
    const h = harness(), res = await h.run(await request())
    expect(res.status).toBe(204); expect(await res.text()).toBe("")
    expect(res.headers.get("cache-control")).toBe("private, no-store")
    expect(h.records).toHaveLength(1)
    expect(h.records[0]?.country).toEqual(Option.some("US"))
    expect(Object.keys(h.records[0]!)).toEqual(["installation", "request", "country", "offeredVersion"])
  })
  it("returns the newest admissible offer regardless of store order", async () => {
    const newest = await signed("3.0.0"), h = harness([await signed("2.0.0"), newest])
    const res = await h.run(await request())
    expect(res.status).toBe(200); expect(await res.json()).toEqual(newest)
    expect(h.records[0]?.offeredVersion).toEqual(Option.some("3.0.0"))
  })
  it("admits a replay only once, including concurrent requests", async () => {
    const h = harness(), req = await request()
    const responses = await Promise.all(Array.from({ length: 10 }, () => h.run(req.clone())))
    expect(responses.filter(res => res.status === 204)).toHaveLength(1)
    expect(responses.filter(res => res.status === 409)).toHaveLength(9)
    expect(h.records).toHaveLength(1)
  })
  it("does not admit unsigned requests or accept other routes/methods", async () => {
    const h = harness(), req = await request()
    expect((await h.run(new Request(req.url))).status).toBe(401)
    expect((await h.run(new Request(req.url, { method: "POST", headers: req.headers }))).status).toBe(405)
    expect((await h.run(new Request(req.url.replace("magnitude.dev", "other.example"), { headers: req.headers }))).status).toBe(404)
    expect(h.records).toHaveLength(0)
  })
  it("does not claim up-to-date during a release-store outage", async () => {
    const h = harness([], { candidates: () => Effect.fail(new DistributionStoreUnavailable()) })
    expect((await h.run(await request())).status).toBe(503)
  })
  it("still serves updates when only telemetry persistence fails", async () => {
    const h = harness([await signed("2.0.0")], { recordCheck: () => Effect.fail(new DistributionStoreUnavailable()) })
    expect((await h.run(await request())).status).toBe(200)
  })
})
