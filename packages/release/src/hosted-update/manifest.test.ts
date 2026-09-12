import { generateKeyPairSync } from "node:crypto"
import { Effect, Either, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { UpdateManifest, PublisherKeyId, signUpdateManifest, verifyUpdateManifest, acceptsUpdateManifest } from "./manifest"
import { UpdateRequest } from "./request"
const keys = generateKeyPairSync("ed25519"), keyId = PublisherKeyId.make("release-2026")
const trust = new Map([[keyId, keys.publicKey]])
const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, path: "releases/2.0.0/magnitude-darwin-arm64.zip", bytes: 100, sha256: "a".repeat(64),
} })
const request = Schema.decodeUnknownSync(UpdateRequest)({ protocol: "1", product: "desktop", version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip", channel: "stable", ts: "100", nonce: "ABCDEFGHIJKLMNOPQRSTUA" })
describe("publisher-signed update manifest", () => {
  it("authenticates exact release bytes and the compatible target", async () => {
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, keyId, keys.privateKey))
    const decoded = await Effect.runPromise(verifyUpdateManifest(envelope, trust))
    expect(decoded).toEqual(manifest)
    expect(acceptsUpdateManifest(decoded, request)).toBe(true)
  })
  it("rejects payload changes, signature changes, unknown publishers, and installation keys", async () => {
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, keyId, keys.privateKey))
    const altered = { ...manifest, version: "3.0.0" }
    for (const input of [{ ...envelope, payload: Buffer.from(JSON.stringify(altered)).toString("base64") }, { ...envelope, signature: "a".repeat(88) }, { ...envelope, keyId: "unknown" }]) {
      expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateManifest(input, trust))))).toBe(true)
    }
    expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateManifest(envelope, new Map([[keyId, generateKeyPairSync("ed25519").publicKey]])))))).toBe(true)
  })
  it("rejects downgrade, prerelease escape, wrong architecture and package", () => {
    for (const candidate of [{ ...manifest, version: "0.9.0" }, { ...manifest, version: "1.0.0" }, { ...manifest, version: "3.0.0-beta.1" }, { ...manifest, artifact: { ...manifest.artifact, target: { ...manifest.artifact.target, arch: "x64" as const } } }]) {
      expect(acceptsUpdateManifest(candidate, request)).toBe(false)
    }
    expect(acceptsUpdateManifest(manifest, { ...request, os: "linux", package: "deb" })).toBe(false)
  })
  it("rejects path traversal and invalid byte counts at the schema boundary", () => {
    for (const artifact of [{ ...manifest.artifact, path: "releases/../secret" }, { ...manifest.artifact, path: "https://untrusted/file" }, { ...manifest.artifact, bytes: -1 }, { ...manifest.artifact, bytes: Number.MAX_SAFE_INTEGER + 1 }]) {
      expect(Either.isLeft(Schema.decodeUnknownEither(UpdateManifest)({ ...manifest, artifact }))).toBe(true)
    }
  })
})
