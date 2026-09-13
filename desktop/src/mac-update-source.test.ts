import { FetchHttpClient } from "@effect/platform"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option, Schema, Stream } from "effect"
import { UpdateClientMetadata, signUpdateRequest } from "@magnitudedev/release/hosted-update"
import { UpdateManifest, PublisherKeyId, signUpdateManifest } from "../../packages/release/src/hosted-update/manifest"
import { UpdatePreferences } from "./update-preferences"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateSource, makeApplicationUpdate } from "./application-update"
import { NativeMacUpdate } from "./mac-update-stage"
import { macUpdateSource } from "./mac-update-source"
import { ApplicationUpdateHandoff } from "./update-handoff"

describe("Mac update acquisition and native handoff", () => {
  it.each([false, true])("verifies downloaded bytes before native staging (corrupt=%s)", async corrupt => {
    const bytes = Buffer.from("verified ZIP fixture")
    const candidate = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" },
      path: "releases/2.0.0/magnitude-desktop-darwin-arm64.zip", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const offer = { manifest: candidate, envelope: await Effect.runPromise(signUpdateManifest(candidate, PublisherKeyId.make("test"), generateKeyPairSync("ed25519").privateKey)) }
    const root = await mkdtemp(join(tmpdir(), "mac-update-source-"))
    const identity = generateKeyPairSync("ed25519")
    const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "darwin", os_version: "26", arch: "arm64", package: "mac-zip" })
    const fetchArtifact = Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init)
      if (new URL(request.url).pathname === "/api/download") {
        expect(request.headers.has("authorization")).toBe(true)
        return new Response(null, { status: 302, headers: { location: `https://storage.example/${candidate.artifact.path}` } })
      }
      expect(request.url).toBe(`https://storage.example/${candidate.artifact.path}`)
      expect(request.headers.has("authorization")).toBe(false)
      return new Response(corrupt ? Buffer.alloc(bytes.length, 0) : bytes, { headers: { "content-length": String(bytes.length) } })
    }, { preconnect: () => {} })
    let staged = false
    let recorded = false
    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const source = yield* macUpdateSource({ origin: "https://magnitude.dev", storageOrigin: "https://storage.example", metadata, sign: url => signUpdateRequest(identity.privateKey, url), trustedPublishers: new Map(), userAgent: "Magnitude/1.0.0", cacheDirectory: root }).pipe(
          Effect.provideService(ApplicationUpdateHandoff, { record: version => Effect.sync(() => { expect(version).toBe(candidate.version); recorded = true }), inspect: () => Effect.succeed({ _tag: "Continue" as const }) }),
          Effect.provideService(NativeMacUpdate, {
            stage: feed => Effect.promise(async () => {
              staged = true
              expect(recorded).toBe(true)
              const manifest = await (await fetch(feed.url, { headers: { Authorization: feed.authorization } })).json() as { url: string }
              expect(Buffer.from(await (await fetch(manifest.url)).arrayBuffer())).toEqual(bytes)
            }),
          }),
        )
        const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, { ...source, check: Effect.succeed(Option.some(offer)) }))
        yield* owner.check
        yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === "Available"), Stream.take(1), Stream.runDrain)
        yield* owner.download
        yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === (corrupt ? "Failed" : "Ready")), Stream.take(1), Stream.runDrain)
        yield* owner.close
      })).pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(false), write: () => Effect.void }), Effect.provideService(FetchHttpClient.Fetch, fetchArtifact), Effect.timeout("5 seconds")))
      expect(staged).toBe(!corrupt)
      expect(recorded).toBe(!corrupt)
      expect(await readdir(root)).toEqual([])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
