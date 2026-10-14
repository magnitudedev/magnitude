import { githubArtifactUrl } from "./github-artifact"
import { FetchHttpClient } from "@effect/platform"
import { Effect, Either, Option, Schema } from "effect"
import { generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { checkHostedUpdate, resolveHostedDownload, UpdateClientMetadata, type UpdateCheck } from "./client"
import { signUpdateRequest, verifyUpdateRequest, installationId } from "./request-auth"
import { decodeUpdateRequest } from "./request"
import { PublisherKeyId, signUpdateManifest, UpdateManifest } from "./manifest"

const installation = generateKeyPairSync("ed25519"), publisher = generateKeyPairSync("ed25519")
const keyId = PublisherKeyId.make("test-publisher")
const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip" })
const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, filename: "app.zip", bytes: 123, sha256: "b".repeat(64),
} })
const check = (fetch: (...args: Parameters<typeof globalThis.fetch>) => Promise<Response>, request: UpdateCheck = { reason: "manual", outcome: Option.none(), hardware: Option.none() }) => checkHostedUpdate({ origin: "https://magnitude.dev", metadata,
  sign: url => signUpdateRequest(installation.privateKey, url), trustedPublishers: new Map([[keyId, publisher.publicKey]]), userAgent: "Magnitude/1.0.0",
}, request).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(fetch, { preconnect: () => {} })))

describe("hosted update client", () => {
  it("signs the download selector and resolves only the trusted GitHub release repository without following it", async () => {
    const expected = githubArtifactUrl(manifest)
    const release = (await Effect.runPromise(signUpdateManifest(manifest, publisher.privateKey))).release
    for (const location of [expected, "https://untrusted.example/file.zip", `${expected}?changed=1`]) {
      let calls = 0
      const result = await Effect.runPromise(resolveHostedDownload({ origin: "https://magnitude.dev", metadata,
        sign: url => signUpdateRequest(installation.privateKey, url), userAgent: "Magnitude/1.0.0", release,
      }).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
        calls++
        const request = new Request(input, init), url = new URL(request.url)
        expect(init?.redirect).toBe("manual")
        expect(url.pathname).toBe("/api/download")
        expect(url.searchParams.has("artifact")).toBe(false)
        expect(url.searchParams.get("release")).toBe(manifest.version)
        expect(await Effect.runPromise(verifyUpdateRequest(request.headers.get("authorization")!, url))).toBe(await Effect.runPromise(installationId(installation.publicKey)))
        return new Response(null, { status: 302, headers: { location } })
      }, { preconnect: () => {} })), Effect.either))
      expect(calls).toBe(1)
      expect(Either.isRight(result)).toBe(location === expected)
    }
  })
  it("carries the check reason and a pending update outcome as signed query fields", async () => {
    const seen: URL[] = []
    const fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init), url = new URL(request.url)
      seen.push(url)
      expect(await Effect.runPromise(verifyUpdateRequest(request.headers.get("authorization")!, url))).toBe(await Effect.runPromise(installationId(installation.publicKey)))
      expect(Option.isSome(await Effect.runPromise(Effect.option(decodeUpdateRequest(url, Math.floor(Date.now() / 1000)))))).toBe(true)
      return new Response(null, { status: 204 })
    }
    const hardware = { backend: "cuda" as const, gpu_name: Option.some("NVIDIA A10-8Q"), gpu_kind: Option.some("discrete" as const), gpu_memory_gb: Option.some(8), gpu_count: Option.some(1),
      gpu_unused_reason: Option.none(), cpu_model: Option.some("AMD EPYC 7V13 64-Core Processor"), cpu_cores: Option.some(12), memory_gb: Option.some(110) }
    await Effect.runPromise(check(fetch, { reason: "launch", outcome: Option.some({ outcome: "failed", version: "1.0.1", from: Option.some("1.0.0"), reason: Option.some("verify") }), hardware: Option.none() }))
    await Effect.runPromise(check(fetch, { reason: "scheduled", outcome: Option.some({ outcome: "applied", version: "1.0.0", from: Option.some("0.9.0"), reason: Option.none() }), hardware: Option.some(hardware) }))
    await Effect.runPromise(check(fetch, { reason: "scheduled", outcome: Option.some({ outcome: "applied", version: "1.0.0", from: Option.none(), reason: Option.none() }), hardware: Option.none() }))
    await Effect.runPromise(check(fetch, { reason: "manual", outcome: Option.none(), hardware: Option.none() }))
    expect(seen.map(url => [url.searchParams.get("reason"), url.searchParams.get("outcome"), url.searchParams.get("outcome_version"), url.searchParams.get("outcome_reason"), url.searchParams.get("outcome_from")])).toEqual([
      ["launch", "failed", "1.0.1", "verify", "1.0.0"], ["scheduled", "applied", "1.0.0", null, "0.9.0"],
      ["scheduled", "applied", "1.0.0", null, null], ["manual", null, null, null, null],
    ])
    expect(seen.map(url => [url.searchParams.get("backend"), url.searchParams.get("gpu_name"), url.searchParams.get("gpu_memory_gb"), url.searchParams.get("cpu_cores"), url.searchParams.has("gpu_unused_reason")])).toEqual([
      [null, null, null, null, false], ["cuda", "NVIDIA A10-8Q", "8", "12", false], [null, null, null, null, false], [null, null, null, null, false],
    ])
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
    const envelope = await Effect.runPromise(signUpdateManifest(manifest, publisher.privateKey))
    const result = await Effect.runPromise(check(async () => Response.json(envelope.release)))
    expect(Option.getOrThrow(result)).toEqual(envelope.release)
    for (const rejected of [{ ...manifest, version: "0.9.0", tag: "@magnitudedev/cli@0.9.0", artifact: { ...manifest.artifact, filename: "app.zip" } }, { ...manifest, artifact: { ...manifest.artifact, target: { ...manifest.artifact.target, arch: "x64" as const } } }]) {
      const offer = await Effect.runPromise(signUpdateManifest(rejected, publisher.privateKey))
      expect(Either.isLeft(await Effect.runPromise(Effect.either(check(async () => Response.json(offer.release)))))).toBe(true)
    }
    expect(Either.isLeft(await Effect.runPromise(Effect.either(check(async () => Response.json({ ...envelope.release, signature: "bad" })))))).toBe(true)
  })
  it.each([301, 302, 401, 409, 429, 500, 503])("does not turn HTTP %s into current or retry it", async status => {
    let calls = 0
    const result = await Effect.runPromise(Effect.either(check(async () => { calls++; return new Response(null, { status, headers: { location: "https://other.invalid" } }) })))
    expect(Either.isLeft(result)).toBe(true)
    expect(calls).toBe(1)
  })
  it("names clock skew when the server refuses a check and its time is more than five minutes from ours", async () => {
    const refusedAt = async (offset: number) => Effect.runPromise(Effect.flip(check(async () =>
      new Response(null, { status: 401, headers: { date: new Date(Date.now() + offset).toUTCString() } }))))
    expect((await refusedAt(2 * 86_400_000)).reason).toBe("clock")
    expect((await refusedAt(-10 * 60_000)).reason).toBe("clock")
    expect((await refusedAt(0)).reason).toBe("response")
  })
  it("bounds the response and rejects malformed JSON", async () => {
    for (const body of ["x".repeat(33000), "not JSON", "{}"]) {
      const result = await Effect.runPromise(Effect.either(check(async () => new Response(body))))
      expect(Either.isLeft(result)).toBe(true)
    }
  })
})
