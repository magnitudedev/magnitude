import { FetchHttpClient } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Option, Schema } from "effect"
import { createHash } from "node:crypto"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { ArtifactDelivery } from "./artifact-delivery"
import { downloadUpdateArtifact } from "./installer-download"
import { UpdateManifest } from "./manifest"

it.each([{ malformedProbe: false, privateDelivery: false }, { malformedProbe: true, privateDelivery: false }, { malformedProbe: true, privateDelivery: true }])("recovers interrupted ranges and incorrect response totals: %j", async ({ malformedProbe, privateDelivery }) => {
  const bytes = Buffer.alloc(9 * 1024 * 1024, 0x61)
  const directory = await mkdtemp(join(tmpdir(), "installer-ranges-"))
  const attempts = new Map<string, number>()
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const range = request.headers.get("range")!
    const match = /^bytes=(\d+)-(\d+)$/.exec(range)
    if (!match) return new Response("bounded requests required", { status: 400 })
    const start = Number(match[1]), end = Number(match[2])
    attempts.set(range, (attempts.get(range) ?? 0) + 1)
    const malformedPart = range === "bytes=0-4194303" && attempts.get(range) === 1
    const malformedBounds = range === "bytes=8388608-9437183" && attempts.get(range) === 1
    const short = start === 4 * 1024 * 1024 && attempts.get(range) === 1
    const body = bytes.subarray(start, short ? start + 1024 : end + 1)
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(body); controller.close() } }), {
      status: 206, headers: { "content-range": `bytes ${start}-${malformedBounds ? end - start : end}/${malformedProbe && range === "bytes=0-0" && attempts.get(range) === 1 ? 1 : malformedPart || malformedBounds ? end - start + 1 : bytes.length}`, etag: '"same-installer"' },
    })
  } })
  try {
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "windows", target: { os: "windows", arch: "x64", package: "windows-exe" }, filename: "app.exe",
      bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const destination = join(directory, "app.exe")
    const result = await Effect.runPromise(downloadUpdateArtifact({ release: { version: manifest.version, bytes: manifest.artifact.bytes, sha256: manifest.artifact.sha256, signature: "A".repeat(86) + "==" }, destination,
      url: privateDelivery ? "https://lab.example/app.exe" : "https://github.com/magnitudedev/magnitude/releases/download/test/app.exe", onProgress: Option.none(),
      artifactDelivery: Schema.decodeUnknownSync(ArtifactDelivery)(privateDelivery ? { _tag: "PrivateAcceptance", origin: "https://lab.example" } : { _tag: "Github" }),
    }).pipe(Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch, Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(init?.redirect).toBe("manual")
      return fetch(`http://127.0.0.1:${server.port}/app.exe`, init)
    }, { preconnect: () => {} }))))
    expect(result.strategy).toBe("Segmented")
    expect(attempts.get("bytes=0-0")).toBe(malformedProbe ? 2 : 1)
    expect(attempts.get("bytes=4194304-8388607")).toBe(2)
    expect(attempts.get("bytes=0-4194303")).toBe(2)
    expect(attempts.get("bytes=8388608-9437183")).toBe(2)
    expect((await readFile(destination)).equals(bytes)).toBe(true)
  } finally { server.stop(true); await rm(directory, { recursive: true, force: true }) }
})

it("does not publish corrupted private acceptance bytes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "private-installer-integrity-"))
  const destination = join(directory, "app.zip")
  const bytes = Buffer.from("tampered package")
  try {
    const result = await Effect.runPromise(downloadUpdateArtifact({
      release: { version: "2.0.0", bytes: bytes.length, sha256: "0".repeat(64), signature: "A".repeat(86) + "==" },
      destination, url: "https://lab.example/app.zip", onProgress: Option.none(),
      artifactDelivery: Schema.decodeUnknownSync(ArtifactDelivery)({ _tag: "PrivateAcceptance", origin: "https://lab.example" }),
    }).pipe(Effect.provide([NodeContext.layer, FetchHttpClient.layer]), Effect.provideService(FetchHttpClient.Fetch,
      Object.assign(async () => new Response(bytes, { headers: { "content-length": String(bytes.length) } }), { preconnect: () => {} })), Effect.either))
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") expect(result.left.phase).toBe("integrity")
    await expect(readFile(destination)).rejects.toMatchObject({ code: "ENOENT" })
  } finally { await rm(directory, { recursive: true, force: true }) }
})
