import { NodeContext } from "@effect/platform-node"
import { FetchHttpClient } from "@effect/platform"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Layer, Option, Schema, Stream } from "effect"
import { UpdateClientMetadata, signUpdateRequest } from "@magnitudedev/release/hosted-update"
import { UpdateManifest, PublisherKeyId, signUpdateManifest } from "../../packages/release/src/hosted-update/manifest"
import { MacUpdateHandoff, UpdatePreferences, makePreparedUpdateStore, PreparedUpdateStore, unixPrivateFilePermissions } from "@magnitudedev/daemon-management/desktop-native"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateSource, makeApplicationUpdate } from "./application-update"
import { NativeMacUpdate } from "./mac-update-stage"
import { macUpdateSource } from "./mac-update-source"
import { installPreparedUpdate, PreparedUpdateInstaller } from "./prepared-update-installation"

describe("Mac update acquisition and native handoff", () => {
  it.each([false, true])("verifies downloaded bytes before native staging (corrupt=%s)", async corrupt => {
    const bytes = Buffer.from("verified ZIP fixture")
    const candidate = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" },
      filename: "magnitude-desktop-darwin-arm64.zip", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const publisher = generateKeyPairSync("ed25519")
    const offer = (await Effect.runPromise(signUpdateManifest(candidate, publisher.privateKey))).release
    const root = await mkdtemp(join(tmpdir(), "mac-update-source-"))
    await writeFile(join(root, "cli"), "helper fixture")
    await writeFile(join(root, "addon"), "addon fixture")
    const identity = generateKeyPairSync("ed25519")
    const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "darwin", os_version: "26", arch: "arm64", package: "mac-zip" })
    const fetchArtifact = Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init)
      if (new URL(request.url).pathname === "/api/download") {
        expect(request.headers.has("authorization")).toBe(true)
        return new Response(null, { status: 302, headers: { location: `https://github.com/magnitudedev/magnitude/releases/download/%40magnitudedev/cli%402.0.0/${candidate.artifact.filename}` } })
      }
      expect(request.url).toBe(`https://github.com/magnitudedev/magnitude/releases/download/%40magnitudedev/cli%402.0.0/${candidate.artifact.filename}`)
      expect(request.headers.has("authorization")).toBe(false)
      return new Response(corrupt ? Buffer.alloc(bytes.length, 0) : bytes, { headers: { "content-length": String(bytes.length) } })
    }, { preconnect: () => {} })
    let staged = false
    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const store = yield* makePreparedUpdateStore({ dataDirectory: root, target: candidate.artifact.target, trustedPublishers: new Map([["test", publisher.publicKey]]) })
        const platform = yield* macUpdateSource({ origin: "https://magnitude.dev", metadata, sign: url => signUpdateRequest(identity.privateKey, url), trustedPublishers: new Map(), userAgent: "Magnitude/1.0.0", cacheDirectory: root, stateDirectory: join(root, "state"), bundle: join(root, "Magnitude.app"), cliPath: join(root, "cli"), addonPath: join(root, "addon") }).pipe(
          Effect.provideService(PreparedUpdateStore, store),
          Effect.provideService(MacUpdateHandoff, { start: () => Effect.succeed({ commit: Effect.sync(() => expect(staged).toBe(true)) }) }),
          Effect.provideService(NativeMacUpdate, {
            stage: feed => Effect.promise(async () => {
              staged = true
              const manifest = await (await fetch(feed.url, { headers: { Authorization: feed.authorization } })).json() as { url: string }
              expect(Buffer.from(await (await fetch(manifest.url)).arrayBuffer())).toEqual(bytes)
            }),
          }),
        )
        const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(PreparedUpdateStore, store), Effect.provideService(ApplicationUpdateSource, { ...platform.source, check: Effect.succeed(Option.some(offer)) }))
        yield* owner.check
        yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === "Available"), Stream.take(1), Stream.runDrain)
        yield* owner.download
        yield* owner.changes.pipe(Stream.filter(state => state.transfer._tag === (corrupt ? "Failed" : "Ready")), Stream.take(1), Stream.runDrain)
        expect(staged).toBe(false)
        yield* owner.close
        if (!corrupt) {
          expect((yield* store.read)._tag).toBe("Some")
          yield* installPreparedUpdate({ showWindow: true, allowAuthorizationPrompt: true }).pipe(Effect.provideService(PreparedUpdateStore, store), Effect.provideService(PreparedUpdateInstaller, platform.installer))
        }
      })).pipe(Effect.provide(unixPrivateFilePermissions.pipe(Layer.provideMerge(NodeContext.layer))), Effect.provideService(UpdatePreferences, { read: Effect.succeed(false), write: () => Effect.void }), Effect.provideService(FetchHttpClient.Fetch, fetchArtifact), Effect.timeout("5 seconds")))
      expect(staged).toBe(!corrupt)
      expect(await readdir(join(root, "updates")).catch(() => [])).toEqual(corrupt ? [] : ["magnitude.zip", "update.json"])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
