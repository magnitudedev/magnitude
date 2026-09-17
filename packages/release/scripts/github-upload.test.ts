import { afterEach, describe, expect, it, vi } from "vitest"
import { Effect } from "effect"
import { ReleaseUploadTransport, UploadHttpError, UploadTransportError, uploadReleaseAssets } from "./github-upload"

const file = { name: "example.tar.gz", path: import.meta.filename, bytes: 32, sha256: "abc" }
const options = {
  repository: "example/project", token: "test", releaseId: 1,
  tag: "0.1.0", sourceCommit: "commit", uploadUrl: "https://uploads.github.com/test",
  files: [file],
}
const uploaded = { id: 2, name: file.name, size: file.bytes, state: "uploaded", digest: "sha256:abc" }
const draft = { draft: true, tag_name: options.tag, target_commitish: options.sourceCommit, assets: [] }

const runUpload = (input: Parameters<typeof uploadReleaseAssets>[0]) => Effect.runPromise(uploadReleaseAssets(input).pipe(
  Effect.provideService(ReleaseUploadTransport, { upload: (url) => Effect.tryPromise({
    try: async () => {
      const response = await fetch(url, { method: "POST" })
      if (!response.ok) throw new UploadHttpError({ status: response.status, message: `HTTP ${response.status}; ${response.headers.get("x-github-request-id")}; ${await response.text()}` })
    },
    catch: error => error instanceof UploadHttpError ? error : new UploadTransportError({ message: String(error) }),
  }) }),
))

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
    await runUpload(options)
    expect(methods.filter(method => method !== "GET")).toEqual(["DELETE", "POST"])
  })

  it("keeps an existing asset only when its state, size and digest match", async () => {
    const fetch = vi.fn(async (url: string, _init: RequestInit) => Response.json(url.includes("/assets?") ? [uploaded] : draft))
    vi.stubGlobal("fetch", fetch)
    await runUpload(options)
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
    await runUpload(options)
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
    await runUpload(options)
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
    await expect(runUpload(options)).rejects.toThrow("request-123; permission denied")
    expect(posts).toBe(1)
  })

  it("refuses to modify a public release or a draft for another source commit", async () => {
    for (const invalid of [{ ...draft, draft: false }, { ...draft, target_commitish: "other" }]) {
      const fetch = vi.fn(async () => Response.json(invalid))
      vi.stubGlobal("fetch", fetch)
      await expect(runUpload(options)).rejects.toThrow("exact private candidate")
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
    await runUpload({ ...options, files: [file, ...firstPage.map(asset => ({ ...file, name: asset.name }))] })
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
    await runUpload(options)
    expect(deleted).toBe(true)
  })

  it("bounds repeated server failures to four attempts", async () => {
    let posts = 0
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      if (init.method === "POST") { posts++; return new Response("unavailable", { status: 503 }) }
      return Response.json(url.includes("/assets?") ? [] : draft)
    }))
    await expect(runUpload(options)).rejects.toThrow("HTTP 503")
    expect(posts).toBe(4)
  }, 20_000)
})

// Exercise the real subprocess rather than mocking curl's response or timeout behavior.
describe("curl upload transport", () => {
  it("streams a large archive with a known length and preserves HTTP failures", async () => {
    const { createServer } = await import("node:http")
    const { mkdtemp, open, rm } = await import("node:fs/promises")
    const { tmpdir } = await import("node:os")
    const { join } = await import("node:path")
    const { CurlReleaseUploadTransport } = await import("./github-upload")
    const directory = await mkdtemp(join(tmpdir(), "upload-transport-test-"))
    const path = join(directory, "large.bin")
    const bytes = 638_320_247
    const handle = await open(path, "w")
    await handle.truncate(bytes)
    await handle.close()
    let received = 0
    let contentLength = ""
    let authorization = ""
    const server = createServer((request, response) => {
      if (request.url === "/error") {
        response.writeHead(500, { "x-github-request-id": "probe-500" }).end("Error saving asset")
        return
      }
      contentLength = request.headers["content-length"] ?? ""
      authorization = request.headers.authorization ?? ""
      request.on("data", chunk => { received += chunk.length })
      request.on("end", () => response.writeHead(201).end("{}"))
    })
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
    const address = server.address() as import("node:net").AddressInfo
    const upload = (route: string) => Effect.runPromise(Effect.flatMap(ReleaseUploadTransport, transport =>
      transport.upload(`http://127.0.0.1:${address.port}/${route}`, "test-token", { ...file, path, bytes }),
    ).pipe(Effect.provide(CurlReleaseUploadTransport)))
    try {
      await upload("success")
      expect({ received, contentLength, authorization }).toEqual({ received: bytes, contentLength: String(bytes), authorization: "Bearer test-token" })
      await expect(upload("error")).rejects.toThrow("probe-500; Error saving asset")
    } finally {
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
      await rm(directory, { recursive: true, force: true })
    }
  }, 60_000)

  it("cuts off a server that accepts bytes but never responds", async () => {
    const { createServer } = await import("node:http")
    const { stat } = await import("node:fs/promises")
    const { CurlReleaseUploadTransport } = await import("./github-upload")
    const server = createServer(request => request.resume())
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve))
    const address = server.address() as import("node:net").AddressInfo
    const bytes = (await stat(file.path)).size
    const started = Date.now()
    try {
      await expect(Effect.runPromise(Effect.flatMap(ReleaseUploadTransport, transport =>
        transport.upload(`http://127.0.0.1:${address.port}/stall`, "test-token", { ...file, bytes }),
      ).pipe(Effect.provide(CurlReleaseUploadTransport)))).rejects.toThrow("curl exit 28")
      expect(Date.now() - started).toBeLessThan(45_000)
    } finally {
      server.closeAllConnections()
      await new Promise<void>(resolve => server.close(() => resolve()))
    }
  }, 50_000)
})
