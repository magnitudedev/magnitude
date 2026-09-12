import { FetchHttpClient } from "@effect/platform"
import { Effect, Either, Option, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { checkHostedUpdate, resolveHostedDownload, UpdateClientMetadata } from "./client"
import { signUpdateRequest, verifyUpdateRequest, installationId } from "./request-auth"
import { PublisherKeyId, signUpdateManifest, UpdateManifest } from "./manifest"

const installation = generateKeyPairSync("ed25519"), publisher = generateKeyPairSync("ed25519")
const keyId = PublisherKeyId.make("test-publisher")
const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip" })
const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, path: "releases/2.0.0/app.zip", bytes: 123, sha256: "b".repeat(64),
} })
const check = (fetch: (...args: Parameters<typeof globalThis.fetch>) => Promise<Response>) => checkHostedUpdate({ origin: "https://magnitude.dev", metadata,
  sign: url => signUpdateRequest(installation.privateKey, url), trustedPublishers: new Map([[keyId, publisher.publicKey]]), userAgent: "Magnitude/1.0.0",
}).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(fetch, { preconnect: () => {} })))

describe("hosted update client", () => {
  it("signs the download selector and resolves only the exact trusted storage path without following it", async () => {
    const expected = `https://storage.example/${manifest.artifact.path}`
    for (const location of [expected, "https://untrusted.example/file.zip", `${expected}?changed=1`]) {
      let calls = 0
      const result = await Effect.runPromise(resolveHostedDownload({ origin: "https://magnitude.dev", metadata,
        sign: url => signUpdateRequest(installation.privateKey, url), userAgent: "Magnitude/1.0.0", manifest, storageOrigin: "https://storage.example",
      }).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
        calls++
        const request = new Request(input, init), url = new URL(request.url)
        expect(init?.redirect).toBe("manual")
        expect(url.pathname).toBe(`/api/download/${manifest.artifact.id}`)
        expect(url.searchParams.get("release")).toBe(manifest.version)
        expect(await Effect.runPromise(verifyUpdateRequest(request.headers.get("authorization")!, url))).toBe(await Effect.runPromise(installationId(installation.publicKey)))
        return new Response(null, { status: 302, headers: { location } })
      }, { preconnect: () => {} })), Effect.either))
      expect(calls).toBe(1)
      expect(Either.isRight(result)).toBe(location === expected)
    }
  })
  it("sends a verifiable complete request and accepts 204 without a body", async () => {
    let calls = 0
    const result = await Effect.runPromise(check(async (input, init) => {
      calls++
      const request = new Request(input, init)
      expect(init?.redirect).toBe("manual")
      expect(request.headers.get("user-agent")).toBe("Magnitude/1.0.0")
      expect(await Effect.runPromise(verifyUpdateRequest(request.headers.get("authorization")!, new URL(request.url))))
        .toBe(await Effect.runPromise(installationId(installation.publicKey)))
      return new Response(null, { status: 204 })
    }))
    expect(Option.isNone(result)).toBe(true)
    expect(calls).toBe(1)
  })
  it("accepts only a matching newer publisher-signed artifact", async () => {
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, keyId, publisher.privateKey))
    const result = await Effect.runPromise(check(async () => Response.json(envelope)))
    expect(Option.getOrThrow(result)).toEqual(manifest)
    for (const rejected of [{ ...manifest, version: "0.9.0", artifact: { ...manifest.artifact, path: "releases/0.9.0/app.zip" } }, { ...manifest, artifact: { ...manifest.artifact, target: { ...manifest.artifact.target, arch: "x64" as const } } }]) {
      const offer = await Effect.runPromise(signUpdateManifest(rejected, keyId, publisher.privateKey))
      expect(Either.isLeft(await Effect.runPromise(Effect.either(check(async () => Response.json(offer)))))).toBe(true)
    }
    expect(Either.isLeft(await Effect.runPromise(Effect.either(check(async () => Response.json({ ...envelope, signature: "bad" })))))).toBe(true)
  })
  it.each([301, 302, 401, 409, 429, 500, 503])("does not turn HTTP %s into current or retry it", async status => {
    let calls = 0
    const result = await Effect.runPromise(Effect.either(check(async () => { calls++; return new Response(null, { status, headers: { location: "https://other.invalid" } }) })))
    expect(Either.isLeft(result)).toBe(true)
    expect(calls).toBe(1)
  })
  it("bounds the response and rejects malformed JSON", async () => {
    for (const body of ["x".repeat(33000), "not JSON", "{}"]) {
      const result = await Effect.runPromise(Effect.either(check(async () => new Response(body))))
      expect(Either.isLeft(result)).toBe(true)
    }
  })
})
