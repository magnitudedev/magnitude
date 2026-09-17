import { afterEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import { uploadReleaseAssets } from "./github-upload"

const file = { name: "example.tar.gz", path: import.meta.filename, bytes: 32, sha256: "abc" }
const options = {
  repository: "example/project", token: "test", releaseId: 1,
  tag: "0.1.0", sourceCommit: "commit", uploadUrl: "https://uploads.github.com/test",
  files: [file],
}
const uploaded = { id: 2, name: file.name, size: file.bytes, state: "uploaded", digest: "sha256:abc" }
const draft = { draft: true, tag_name: options.tag, target_commitish: options.sourceCommit, assets: [] }

afterEach(() => vi.unstubAllGlobals())

describe("release upload recovery", () => {
  it("finds and removes starter assets omitted from the release response", async () => {
    let assets = [{ ...uploaded, state: "starter" }]
    const methods: string[] = []
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      methods.push(init.method ?? "GET")
      if (init.method === "DELETE") { assets = []; return new Response(null, { status: 204 }) }
      if (init.method === "POST") return Response.json(uploaded, { status: 201 })
      return Response.json(url.includes("/assets?") ? assets : draft)
    }))
    await Effect.runPromise(uploadReleaseAssets(options))
    expect(methods.filter(method => method !== "GET")).toEqual(["DELETE", "POST"])
  })

  it("keeps an existing asset only when its state, size and digest match", async () => {
    const fetch = vi.fn(async (url: string, _init: RequestInit) => Response.json(url.includes("/assets?") ? [uploaded] : draft))
    vi.stubGlobal("fetch", fetch)
    await Effect.runPromise(uploadReleaseAssets(options))
    expect(fetch.mock.calls.every(([, init]) => !init.method)).toBe(true)
  })

  it("reconciles a 500 that left a starter before retrying just that file", async () => {
    let assets: Array<typeof uploaded> = []
    let posts = 0
    let deletes = 0
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "DELETE") { deletes++; assets = []; return new Response(null, { status: 204 }) }
      if (init.method === "POST") {
        posts++
        if (posts === 1) { assets = [{ ...uploaded, state: "starter" }]; return new Response("upstream failure", { status: 500 }) }
        return Response.json(uploaded, { status: 201 })
      }
      return Response.json(url.includes("/assets?") ? assets : draft)
    }))
    await Effect.runPromise(uploadReleaseAssets(options))
    expect({ posts, deletes }).toEqual({ posts: 2, deletes: 1 })
  })

  it("does not repeat an upload stored successfully despite a lost response", async () => {
    let assets: Array<typeof uploaded> = []
    let posts = 0
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "POST") { posts++; assets = [uploaded]; throw new Error("connection reset") }
      if (init.method === "DELETE") throw new Error("must not delete a verified upload")
      return Response.json(url.includes("/assets?") ? assets : draft)
    }))
    await Effect.runPromise(uploadReleaseAssets(options))
    expect(posts).toBe(1)
  })

  it("does not retry authorization failures and retains GitHub diagnostics", async () => {
    let posts = 0
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "POST") {
        posts++
        return new Response("permission denied", { status: 403, headers: { "x-github-request-id": "request-123" } })
      }
      return Response.json(url.includes("/assets?") ? [] : draft)
    }))
    await expect(Effect.runPromise(uploadReleaseAssets(options))).rejects.toThrow("request-123; permission denied")
    expect(posts).toBe(1)
  })

  it("refuses to modify a public release or a draft for another source commit", async () => {
    for (const invalid of [{ ...draft, draft: false }, { ...draft, target_commitish: "other" }]) {
      const fetch = vi.fn(async () => Response.json(invalid))
      vi.stubGlobal("fetch", fetch)
      await expect(Effect.runPromise(uploadReleaseAssets(options))).rejects.toThrow("exact private candidate")
      expect(fetch).toHaveBeenCalledTimes(1)
    }
  })

  it("reads later asset pages and removes a hidden starter there", async () => {
    const firstPage = Array.from({ length: 100 }, (_, id) => ({ ...uploaded, id: id + 10, name: `existing-${id}` }))
    let starter = [{ ...uploaded, state: "starter" }]
    let removed = false
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "DELETE") { removed = true; starter = []; return new Response(null, { status: 204 }) }
      if (init.method === "POST") return Response.json(uploaded, { status: 201 })
      if (url.includes("/assets?")) return Response.json(url.endsWith("page=1") ? firstPage : starter)
      return Response.json(draft)
    }))
    await Effect.runPromise(uploadReleaseAssets({ ...options, files: [file, ...firstPage.map(asset => ({ ...file, name: asset.name }))] }))
    expect(removed).toBe(true)
  })

  it("replaces an uploaded asset whose checksum differs", async () => {
    let assets = [{ ...uploaded, digest: "sha256:wrong" }]
    let deleted = false
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "DELETE") { deleted = true; assets = []; return new Response(null, { status: 204 }) }
      if (init.method === "POST") return Response.json(uploaded, { status: 201 })
      return Response.json(url.includes("/assets?") ? assets : draft)
    }))
    await Effect.runPromise(uploadReleaseAssets(options))
    expect(deleted).toBe(true)
  })

  it("bounds repeated server failures to four attempts", async () => {
    let posts = 0
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "POST") { posts++; return new Response("unavailable", { status: 503 }) }
      return Response.json(url.includes("/assets?") ? [] : draft)
    }))
    await expect(Effect.runPromise(uploadReleaseAssets(options))).rejects.toThrow("HTTP 503")
    expect(posts).toBe(4)
  }, 20_000)
})
