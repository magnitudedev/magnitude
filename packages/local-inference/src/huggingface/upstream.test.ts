import * as BunContext from "@effect/platform-bun/BunContext"
import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { Effect, Option } from "effect"
import { describe, expect, it, vi } from "vitest"
import { makeHuggingFaceUpstream } from "./upstream"

describe("@huggingface/hub cache compatibility", () => {
  it("uses the official incomplete/blob/snapshot publication flow in a custom cache root", async () => {
    const debug = vi.spyOn(console, "debug").mockImplementation(() => {})
    const bytes = new TextEncoder().encode("abcdefgh")
    const oid = "b".repeat(64)
    const commit = "a".repeat(40)
    const customFetch = (async (input: URL | RequestInfo, init?: RequestInit) => {
      const url = String(input)
      if (url.includes("/paths-info/")) {
        return Response.json([{
          path: "model.gguf",
          type: "file",
          size: bytes.length,
          oid: "c".repeat(40),
          lfs: { oid, size: bytes.length, pointerSize: 120 },
          lastCommit: { id: commit, title: "fixture", date: new Date().toISOString() },
          securityFileStatus: { status: "safe" },
        }])
      }
      const headers = new Headers(init?.headers)
      if (headers.get("range") === "bytes=0-0") {
        return new Response(bytes.slice(0, 1), { status: 206, headers: { "content-range": `bytes 0-0/${bytes.length}`, etag: oid } })
      }
      return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length), etag: oid } })
    }) as typeof fetch

    const result = await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      const path = yield* Path.Path
      const temporary = yield* fs.makeTempDirectoryScoped()
      const cacheDir = path.join(temporary, "cache")
      const upstream = makeHuggingFaceUpstream({ hubUrl: Option.some(new URL("https://hub.test")), token: Option.none(), fetch: Option.some(customFetch) })
      const pointer = yield* upstream.downloadToCache({ repository: "owner/model", commit, path: "model.gguf", cacheDir })
      return {
        pointer,
        pointerBytes: yield* fs.readFile(pointer),
        blobExists: yield* fs.exists(path.join(cacheDir, "models--owner--model", "blobs", oid)),
        incompleteExists: yield* fs.exists(path.join(cacheDir, "models--owner--model", "blobs", `${oid}.incomplete`)),
      }
    }).pipe(Effect.provide(BunContext.layer))))

    expect(result.pointer).toContain(`/snapshots/${commit}/model.gguf`)
    expect([...result.pointerBytes]).toEqual([...bytes])
    expect(result.blobExists).toBe(true)
    expect(result.incompleteExists).toBe(false)
    debug.mockRestore()
  })
})
