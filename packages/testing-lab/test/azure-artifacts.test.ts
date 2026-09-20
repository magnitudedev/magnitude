import { FetchHttpClient, HttpClient, HttpClientResponse } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Clock, Effect, Layer, Schema, Stream, TestClock, TestContext } from "effect"
import { expect, test } from "vitest"
import { ArtifactStore } from "../src/artifact-store"
import { azureArtifactStore } from "../src/providers/azure-artifacts"
import { ProcessExecutor } from "../src/process"
import { sha256 } from "../src/snapshot"

const config = { executable: "az", subscription: "5304c4b3-d605-4193-b0cb-766c065acfa6", account: "fixtureaccount", container: "artifacts", maxBytes: 1024 }
const json = Schema.encodeSync(Schema.parseJson(Schema.Unknown))
for (const mode of ["valid", "empty", "corrupt-download", "oversized", "changed", "concurrent-writer"] as const) test(`Azure HTTP artifacts preserve immutable byte admission: ${mode}`, () => Effect.runPromise(Effect.gen(function* () {
  const bytes = new TextEncoder().encode(mode === "empty" ? "" : "private candidate bytes"), digest = sha256(bytes)
  const methods: string[] = []
  let uploaded = false, reading = false, credentials = 0
  const transport = HttpClient.make(request => Effect.gen(function* () {
    methods.push(request.method)
    expect(request.url).toBe(`https://fixtureaccount.blob.core.windows.net/artifacts/${digest}`)
    expect(request.headers.authorization).toBe("Bearer fixture-secret")
    const length = reading && mode === "oversized" ? 2048 : bytes.length
    const headers = { "content-length": String(length), etag: '"fixture-etag"' }
    if (request.method === "HEAD") return HttpClientResponse.fromWeb(request, new Response(null, { status: uploaded ? 200 : 404, headers }))
    if (request.method === "PUT") {
      expect(request.headers["if-none-match"]).toBe("*")
      expect(request.headers["x-ms-blob-type"]).toBe("BlockBlob")
      expect(request.body.contentLength).toBe(bytes.length)
      expect(request.headers["content-length"]).toBe(String(bytes.length))
      if (request.body._tag === "Uint8Array") expect(sha256(request.body.body)).toBe(digest)
      else if (request.body._tag === "Stream") expect(sha256(Buffer.concat(Array.from(yield* request.body.stream.pipe(Stream.runCollect, Effect.orDie))))).toBe(digest)
      else return yield* Effect.dieMessage("Expected an empty body or streaming upload")
      uploaded = true
      return HttpClientResponse.fromWeb(request, new Response(null, { status: mode === "concurrent-writer" ? 412 : 201 }))
    }
    expect(request.headers["if-match"]).toBe('"fixture-etag"')
    if (mode === "changed") return HttpClientResponse.fromWeb(request, new Response(null, { status: 412 }))
    const received = bytes.slice()
    if (mode === "corrupt-download") received[0] ^= 255
    return HttpClientResponse.fromWeb(request, new Response(received, { headers }))
  }))
  const program = Effect.gen(function* () {
    const store = yield* ArtifactStore
    expect((yield* store.put(digest, Stream.make(new Uint8Array([0]))).pipe(Effect.either))._tag).toBe("Left")
    expect(methods).toEqual([])
    expect(credentials).toBe(0)
    yield* store.put(digest, Stream.make(bytes))
    yield* store.put(digest, Stream.make(bytes))
    expect(methods.filter(method => method === "PUT")).toHaveLength(1)
    expect(yield* store.exists(digest)).toBe(true)
    reading = true
    const downloaded = yield* store.get(digest).pipe(Stream.runCollect, Effect.either)
    expect(downloaded._tag).toBe(mode === "valid" || mode === "empty" || mode === "concurrent-writer" ? "Right" : "Left")
    if (downloaded._tag === "Right") expect(Buffer.concat(Array.from(downloaded.right))).toEqual(Buffer.from(bytes))
    if (mode === "oversized") expect(methods).not.toContain("GET")
    expect(credentials).toBe(1)
  }).pipe(Effect.provide(azureArtifactStore(config)))
  yield* program.pipe(Effect.provideService(HttpClient.HttpClient, transport), Effect.provideService(ProcessExecutor, { run: spec => Effect.sync(() => {
    expect(spec.args).toEqual(["account", "get-access-token", "--subscription", config.subscription, "--resource", "https://storage.azure.com/", "--only-show-errors", "--output", "json"])
    credentials++
    return { exitCode: 0, stdout: json({ accessToken: "fixture-secret", expires_on: Math.floor(Date.now() / 1000) + 3600 }), stderr: "" }
  }) }))
}).pipe(Effect.provide(BunContext.layer))))

test("concurrent object requests share a renewable credential and auth failures never look like missing objects", () => Effect.runPromise(Effect.gen(function* () {
  let issued = 0, denied = false
  const transport = Layer.succeed(HttpClient.HttpClient, HttpClient.make(request => Effect.sync(() => {
    expect(request.headers.authorization).toBe(`Bearer token-${issued}`)
    return HttpClientResponse.fromWeb(request, new Response(null, { status: denied ? 403 : 404 }))
  })))
  const executor = Layer.succeed(ProcessExecutor, { run: () => Effect.gen(function* () {
    const now = yield* Clock.currentTimeMillis
    issued++
    yield* Effect.yieldNow()
    return { exitCode: 0, stdout: json({ accessToken: `token-${issued}`, expires_on: Math.floor(now / 1000) + 300 }), stderr: "" }
  }) })
  yield* Effect.gen(function* () {
    const store = yield* ArtifactStore
    expect(yield* Effect.all(Array.from({ length: 50 }, (_, i) => store.exists(sha256(String(i)))), { concurrency: 10 })).toEqual(Array(50).fill(false))
    expect(issued).toBe(1)
    yield* TestClock.adjust("181 seconds")
    expect(yield* store.exists(sha256("renew"))).toBe(false)
    expect(issued).toBe(2)
    denied = true
    expect(yield* store.exists(sha256("private")).pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { message: "Azure blob metadata returned HTTP 403" } })
  }).pipe(Effect.provide(azureArtifactStore(config).pipe(Layer.provide(Layer.merge(transport, executor)))))
}).pipe(Effect.provide(Layer.merge(BunContext.layer, TestContext.TestContext)))))

test("the actual fetch adapter disables redirects before attaching Azure authorization", () => Effect.runPromise(Effect.gen(function* () {
  let calls = 0
  const request = Effect.flatMap(ArtifactStore, store => store.exists(sha256("redirect"))).pipe(
    Effect.provide(azureArtifactStore(config).pipe(Layer.provide(FetchHttpClient.layer))),
    Effect.provideService(FetchHttpClient.Fetch, Object.assign(async (_url: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
      calls++
      expect(init?.redirect).toBe("manual")
      return new Response(null, { status: 302, headers: { location: "https://unrelated.example/" } })
    }, { preconnect: () => {} })),
    Effect.provideService(ProcessExecutor, { run: () => Effect.succeed({ exitCode: 0,
      stdout: json({ accessToken: "fixture-secret", expires_on: Math.floor(Date.now() / 1000) + 3600 }), stderr: "" }) }),
  )
  expect(yield* request.pipe(Effect.either)).toMatchObject({ _tag: "Left", left: { message: "Azure blob metadata returned HTTP 302" } })
  expect(calls).toBe(1)
}).pipe(Effect.provide(BunContext.layer))))
