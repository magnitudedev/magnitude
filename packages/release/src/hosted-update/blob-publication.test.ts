import { FileSystem, HttpClient, HttpClientResponse } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { createHash } from "node:crypto"
import { Effect, Schema } from "effect"
import { join } from "node:path"
import { describe, expect, it, vi } from "vitest"
import { publishArtifactBytes } from "./blob-publication"
import { UpdateManifest } from "./manifest"

vi.mock("@vercel/blob", () => ({ head: vi.fn(async () => ({})), put: vi.fn(async () => { throw new Error("Existing artifacts must not be overwritten") }), BlobNotFoundError: class extends Error {} }))
const bytes = Buffer.from("the complete artifact")
const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40),
  artifact: { id: "mac", target: { os: "darwin", arch: "arm64", package: "mac-zip" }, path: "releases/2.0.0/mac.zip", bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") } })
const run = (response: () => Response) => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = yield* fs.makeTempDirectoryScoped({ prefix: "publication-bytes-" })
  const file = join(directory, "archive.zip")
  yield* fs.writeFile(file, bytes)
  return yield* publishArtifactBytes({ file, manifest, token: "test", storageOrigin: "https://downloads.magnitude.dev" }).pipe(
    Effect.provideService(HttpClient.HttpClient, HttpClient.make(request => Effect.succeed(HttpClientResponse.fromWeb(request, response())))), Effect.either)
})).pipe(Effect.provide(NodeContext.layer)))

describe("immutable artifact remote verification", () => {
  it("revalidates the complete bytes of an existing object", async () => {
    expect((await run(() => new Response(bytes)))).toMatchObject({ _tag: "Right", right: "https://downloads.magnitude.dev/releases/2.0.0/mac.zip" })
  })
  it.each([
    [() => new Response(bytes, { status: 503 }), "HTTP 503"],
    [() => new Response(bytes.subarray(0, 4)), "contained 4 bytes"],
    [() => new Response(Buffer.alloc(bytes.length)), "digest does not match"],
    [() => new Response(Buffer.concat([bytes, bytes])), "exceeded its expected"],
  ] as const)("rejects an invalid remote object with useful diagnostics", async (response, message) => {
    const result = await run(response)
    expect(result._tag).toBe("Left")
    if (result._tag === "Left") {
      expect(result.left.stage).toBe("remote-verification")
      expect(result.left.message).toContain(message)
    }
  })
})
