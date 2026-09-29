import { FetchHttpClient } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Either, Option, Schema } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { checkHostedUpdate, resolveHostedDownload, UpdateClientMetadata } from "./client"
import { downloadUpdateArtifact } from "./installer-download"
import { githubArtifactUrl } from "./github-artifact"
import { signUpdateManifest, UpdateManifest } from "./manifest"
import { signUpdateRequest, verifyUpdateRequest } from "./request-auth"

it("checks a signed hosted offer and verifies the selected release asset before use", async () => {
  const installation = generateKeyPairSync("ed25519")
  const publisher = generateKeyPairSync("ed25519")
  const bytes = Buffer.from("isolated hosted update fixture\n")
  const manifest = Schema.decodeUnknownSync(UpdateManifest)({
    protocol: 1, version: "1.0.1", tag: "@magnitudedev/cli@1.0.1", commit: "a".repeat(40),
    artifact: { id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" },
      filename: "magnitude-desktop-darwin-arm64.zip", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex") },
  })
  const publication = await Effect.runPromise(signUpdateManifest(manifest, publisher.privateKey))
  const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "darwin", os_version: "26.0", arch: "arm64", package: "mac-zip" })
  const connection = { origin: "https://magnitude.dev", metadata,
    sign: (url: URL) => signUpdateRequest(installation.privateKey, url),
    trustedPublishers: new Map([["fixture", publisher.publicKey]]), userAgent: "Magnitude/1.0.0" }
  const artifactUrl = githubArtifactUrl(manifest)
  const directory = await mkdtemp(join(tmpdir(), "hosted-update-flow-"))
  const requests: string[] = []
  let corrupt = false
  const localFetch = Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
    const request = new Request(input, init)
    const url = new URL(request.url)
    requests.push(url.pathname)
    expect(init?.redirect).toBe("manual")
    if (url.origin === connection.origin) {
      expect(await Effect.runPromise(verifyUpdateRequest(request.headers.get("authorization")!, url))).toBeTruthy()
      if (url.pathname === "/api/update") return Response.json(publication.release)
      expect(url.pathname).toBe("/api/download")
      expect(url.searchParams.get("release")).toBe(manifest.version)
      return new Response(null, { status: 302, headers: { location: artifactUrl } })
    }
    expect(url.href).toBe(artifactUrl)
    expect(request.headers.has("authorization")).toBe(false)
    const payload = corrupt ? Buffer.from("x".repeat(bytes.length)) : bytes
    const range = /^bytes=(\d+)-(\d+)$/.exec(request.headers.get("range") ?? "")
    if (range) {
      const start = Number(range[1]), end = Number(range[2])
      return new Response(payload.subarray(start, end + 1), { status: 206,
        headers: { "content-range": `bytes ${start}-${end}/${payload.length}`, etag: '"fixture"' } })
    }
    return new Response(payload, { headers: { "content-length": String(payload.length) } })
  }, { preconnect: () => {} })
  try {
    const offer = Option.getOrThrow(await Effect.runPromise(checkHostedUpdate(connection, { reason: "manual", outcome: Option.none() }).pipe(
      Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch, localFetch))))
    expect(offer).toEqual(publication.release)
    const selected = await Effect.runPromise(resolveHostedDownload({ ...connection, release: offer }).pipe(
      Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch, localFetch)))
    expect(selected).toBe(artifactUrl)
    const destination = join(directory, "update.zip")
    await Effect.runPromise(downloadUpdateArtifact({ release: offer, url: selected, destination, onProgress: Option.none() }).pipe(
      Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch, localFetch)))
    expect(await readFile(destination)).toEqual(bytes)

    corrupt = true
    const rejected = await Effect.runPromise(downloadUpdateArtifact({ release: offer, url: selected,
      destination: join(directory, "corrupt.zip"), onProgress: Option.none() }).pipe(Effect.either,
      Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch, localFetch)))
    expect(Either.isLeft(rejected)).toBe(true)
    expect(requests).toContain("/api/update")
    expect(requests).toContain("/api/download")
    expect(requests).toContain(new URL(artifactUrl).pathname)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
