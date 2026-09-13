import { FetchHttpClient } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Option, Schema } from "effect"
import { createHash } from "node:crypto"
import { mkdtemp, readFile, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { expect, it } from "vitest"
import { downloadUpdateArtifact } from "./installer-download"
import { UpdateManifest } from "./manifest"

it.each([false, true])("recovers interrupted ranges with malformed initial negotiation=%s", async malformedProbe => {
  const bytes = Buffer.alloc(9 * 1024 * 1024, 0x61)
  const directory = await mkdtemp(join(tmpdir(), "installer-ranges-"))
  const attempts = new Map<string, number>()
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch(request) {
    const range = request.headers.get("range")!
    const match = /^bytes=(\d+)-(\d+)$/.exec(range)
    if (!match) return new Response("bounded requests required", { status: 400 })
    const start = Number(match[1]), end = Number(match[2])
    attempts.set(range, (attempts.get(range) ?? 0) + 1)
    const short = start === 4 * 1024 * 1024 && attempts.get(range) === 1
    const body = bytes.subarray(start, short ? start + 1024 : end + 1)
    return new Response(new ReadableStream({ start(controller) { controller.enqueue(body); controller.close() } }), {
      status: 206, headers: { "content-range": `bytes ${start}-${end}/${malformedProbe && range === "bytes=0-0" && attempts.get(range) === 1 ? 1 : bytes.length}`, etag: '"same-installer"' },
    })
  } })
  try {
    const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
      id: "windows", target: { os: "windows", arch: "x64", package: "windows-exe" }, path: "releases/2.0.0/app.exe",
      bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const destination = join(directory, "app.exe")
    const result = await Effect.runPromise(downloadUpdateArtifact({ manifest, destination,
      url: `http://127.0.0.1:${server.port}/app.exe`, onProgress: Option.none(),
    }).pipe(Effect.provide([NodeContext.layer, FetchHttpClient.layer])))
    expect(result.strategy).toBe("Segmented")
    expect(attempts.get("bytes=0-0")).toBe(malformedProbe ? 2 : 1)
    expect(attempts.get("bytes=4194304-8388607")).toBe(2)
    expect(attempts.get("bytes=0-4194303")).toBe(1)
    expect(attempts.get("bytes=8388608-9437183")).toBe(1)
    expect((await readFile(destination)).equals(bytes)).toBe(true)
  } finally { server.stop(true); await rm(directory, { recursive: true, force: true }) }
})
