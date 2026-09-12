import { createPrivateKey, generateKeyPairSync } from "node:crypto"
import { Effect, Either } from "effect"
import { describe, expect, it } from "vitest"
import { installationId, newUpdateNonce, signUpdateRequest, updateQuery, verifyUpdateRequest } from "./request-auth"
import { decodeUpdateRequest } from "./request"
import goVector from "./go-ssh-vector.json"

const pair = generateKeyPairSync("ed25519")
const url = () => new URL(`https://magnitude.dev/api/update?${updateQuery({ protocol: "1", product: "desktop", version: "1.2.3", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip", channel: "stable", ts: "1000000", nonce: "ABCDEFGHIJKLMNOPQRSTUA" })}`)
describe("Ollama-compatible update authentication", () => {
  it("matches a real Go crypto/ssh signer byte for byte", async () => {
    // Golden vector generated with Go's ssh.NewSignerFromKey and Sign over GET,<RequestURI>.
    // The all-zero seed is a public test fixture, never an installation or publisher key.
    const privateKey = createPrivateKey({ key: Buffer.concat([Buffer.from("302e020100300506032b657004220420", "hex"), Buffer.alloc(32)]), format: "der", type: "pkcs8" })
    const url = new URL(goVector.url)
    expect(await Effect.runPromise(signUpdateRequest(privateKey, url))).toBe(goVector.authorization)
    expect(await Effect.runPromise(verifyUpdateRequest(goVector.authorization, url))).toBe(await Effect.runPromise(installationId(privateKey)))
  })
  it("round-trips an installation identity without exposing private material", async () => {
    const signed = await Effect.runPromise(signUpdateRequest(pair.privateKey, url()))
    expect(await Effect.runPromise(verifyUpdateRequest(signed, url()))).toBe(await Effect.runPromise(installationId(pair.publicKey)))
    expect(signed.split(":").map(value => Buffer.from(value, "base64").length)).toEqual([51, 64])
  })
  it("binds every query field and the request path", async () => {
    const original = url(), signed = await Effect.runPromise(signUpdateRequest(pair.privateKey, original))
    for (const field of original.searchParams.keys()) {
      const changed = new URL(original); changed.searchParams.set(field, "changed")
      expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateRequest(signed, changed))))).toBe(true)
    }
    const changed = new URL(original); changed.pathname = "/api/download/file"
    expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateRequest(signed, changed))))).toBe(true)
  })
  it("rejects malformed, noncanonical and oversized authorization", async () => {
    const signed = await Effect.runPromise(signUpdateRequest(pair.privateKey, url()))
    for (const value of ["", "x:y", signed + ":x", signed.replace(":", " :"), "a".repeat(1000)]) {
      expect(Either.isLeft(await Effect.runPromise(Effect.either(verifyUpdateRequest(value, url()))))).toBe(true)
    }
  })
  it("matches Go query escaping and generates unique 16-byte nonces", async () => {
    expect(updateQuery({ z: "a b+c!()*'~", a: "x/y" })).toBe("a=x%2Fy&z=a+b%2Bc%21%28%29%2A%27~")
    const nonces = await Effect.runPromise(Effect.all(Array.from({ length: 100 }, () => newUpdateNonce)))
    expect(new Set(nonces).size).toBe(100)
    expect(nonces.every(value => Buffer.from(value, "base64url").length === 16)).toBe(true)
  })
})
describe("update request admission", () => {
  it("accepts the clock-skew boundary and rejects an expired request", async () => {
    expect((await Effect.runPromise(decodeUpdateRequest(url(), 1000300))).version).toBe("1.2.3")
    expect(Either.isLeft(await Effect.runPromise(Effect.either(decodeUpdateRequest(url(), 1000301))))).toBe(true)
  })
  it("rejects duplicate, unknown, incompatible, or malformed fields", async () => {
    for (const [key, value] of [["version", "bad"], ["package", "rpm"], ["ts", "NaN"], ["nonce", "short"], ["unknown", "x"]]) {
      const changed = url(); changed.searchParams.set(key!, value!)
      expect(Either.isLeft(await Effect.runPromise(Effect.either(decodeUpdateRequest(changed, 1000000))))).toBe(true)
    }
    const changed = url(); changed.searchParams.append("version", "1.2.3")
    expect(Either.isLeft(await Effect.runPromise(Effect.either(decodeUpdateRequest(changed, 1000000))))).toBe(true)
  })
})
