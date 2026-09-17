import { generateKeyPairSync, sign } from "node:crypto"
import { Effect, Either } from "effect"
import { describe, expect, it } from "vitest"
import { acceptsUpdateRelease, signUpdateRelease, verifyUpdateRelease, type ReleaseTarget } from "./release"

const publisher = generateKeyPairSync("ed25519")
const trust = new Map([["current", publisher.publicKey]])
const target: ReleaseTarget = { os: "darwin", arch: "arm64", package: "mac-zip" }
const content = { version: "2.0.0", bytes: 123456789, sha256: "a".repeat(64) }
const signed = () => Effect.runPromise(signUpdateRelease(content, target, publisher.privateKey))
const rejects = async (input: unknown, destination: ReleaseTarget = target) =>
  expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateRelease(input, destination, trust))))).toBe(true)

describe("public update release", () => {
  it("has four fields and signs a fixed domain-separated representation independent of JSON order", async () => {
    const release = await signed()
    expect(Object.keys(release).sort()).toEqual(["bytes", "sha256", "signature", "version"])
    const bytes = Buffer.from(`magnitude-update-release-v1\ndarwin\narm64\nmac-zip\n2.0.0\n123456789\n${content.sha256}\n`)
    expect(release.signature).toBe(sign(null, bytes, publisher.privateKey).toString("base64"))
    expect(await Effect.runPromise(verifyUpdateRelease({ signature: release.signature, sha256: release.sha256, bytes: release.bytes, version: release.version }, target, trust))).toEqual(release)
  })
  it("binds every target and content field", async () => {
    const release = await signed()
    for (const changed of [{ version: "3.0.0" }, { bytes: 1 }, { sha256: "b".repeat(64) }]) await rejects({ ...release, ...changed })
    await rejects(release, { os: "darwin", arch: "x64", package: "mac-zip" })
    await rejects(release, { os: "darwin", arch: "arm64", package: "dmg" })
    await rejects(release, { os: "linux", arch: "arm64", package: "deb" })
  })
  it("tries bundled keys without accepting a key selector or embedded trust", async () => {
    const release = await signed()
    const other = generateKeyPairSync("ed25519")
    const rotated = new Map([["old", other.publicKey], ["new-name", publisher.publicKey]])
    expect(await Effect.runPromise(verifyUpdateRelease(release, target, rotated))).toEqual(release)
    for (const keys of [new Map(), new Map([["current", other.publicKey]])]) {
      expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateRelease(release, target, keys))))).toBe(true)
    }
    await rejects({ ...release, keyId: "current" })
    await rejects({ ...release, publicKey: publisher.publicKey.export({ type: "spki", format: "pem" }) })
  })
  it("rejects malformed signatures, unsafe sizes and the old envelope", async () => {
    const release = await signed()
    for (const signature of ["", "A".repeat(88), release.signature.slice(0, -2), ` ${release.signature}`, "A".repeat(86) + "=="]) await rejects({ ...release, signature })
    for (const bytes of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1, "123456789"]) await rejects({ ...release, bytes })
    await rejects({ keyId: "current", payload: Buffer.from(JSON.stringify(content)).toString("base64"), signature: release.signature })
  })
  it("keeps channel and downgrade admission independent of signature verification", async () => {
    const release = await signed()
    expect(acceptsUpdateRelease(release, "1.0.0")).toBe(true)
    expect(acceptsUpdateRelease(release, "2.0.0")).toBe(false)
    expect(acceptsUpdateRelease(release, "3.0.0")).toBe(false)
    expect(acceptsUpdateRelease({ ...release, version: "3.0.0-beta.1" }, "2.0.0")).toBe(false)
    expect(acceptsUpdateRelease({ ...release, version: "3.0.0-beta.1" }, "2.0.0-alpha.1")).toBe(true)
  })
})
