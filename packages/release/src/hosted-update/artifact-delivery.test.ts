import { FetchHttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Either, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { acceptsArtifactUrl, ArtifactDelivery, artifactDeliveryClient } from "./artifact-delivery"

const policy = Schema.decodeUnknownSync(ArtifactDelivery)({ _tag: "PrivateAcceptance", origin: "https://lab.example:8443" })
const asset = "https://lab.example:8443/artifacts/app.zip?sig=scoped"
const transfer = (url: string, fetcher: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) => Effect.runPromise(Effect.gen(function* () {
  const client = yield* artifactDeliveryClient(policy)
  return yield* client.execute(HttpClientRequest.get(url, { headers: {
    authorization: "installation-secret", cookie: "private", "x-api-key": "secret", range: "bytes=0-0", "if-range": '"etag"',
  } }))
}).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(fetcher, { preconnect: () => {} })), Effect.either))

describe("private acceptance artifact delivery", () => {
  it("uses only the fixed origin and byte-transfer headers", async () => {
    const result = await transfer(asset, async (input, init) => {
      const request = new Request(input, init)
      expect(request.url).toBe(asset)
      expect(init?.redirect).toBe("manual")
      expect(init?.credentials).toBe("omit")
      for (const name of ["authorization", "cookie", "x-api-key"]) expect(request.headers.has(name)).toBe(false)
      expect(request.headers.get("range")).toBe("bytes=0-0")
      expect(request.headers.get("if-range")).toBe('"etag"')
      return new Response("x", { status: 206 })
    })
    expect(Either.isRight(result)).toBe(true)
  })
  it.each([301, 302, 303, 304, 307, 308])("rejects HTTP %s without following even a same-origin redirect", async status => {
    let calls = 0
    const result = await transfer(asset, async () => {
      calls++
      return new Response(null, { status, headers: { location: "https://lab.example:8443/other" } })
    })
    expect(Either.isLeft(result)).toBe(true)
    expect(calls).toBe(1)
  })
  it.each([
    "http://lab.example:8443/file", "https://lab.example/file", "https://lab.example.evil:8443/file",
    "https://user:secret@lab.example:8443/file", "https://lab.example:8443/file#fragment",
    "https://github.com/magnitudedev/magnitude/releases/download/test/file", "not a URL",
  ])("rejects destinations before transport: %s", async url => {
    let calls = 0
    expect(acceptsArtifactUrl(policy, url)).toBe(false)
    expect(Either.isLeft(await transfer(url, async () => { calls++; return new Response("unexpected") }))).toBe(true)
    expect(calls).toBe(0)
  })
  it.each([
    "http://localhost:1234", "https://lab.example/", "https://lab.example/path", "https://user:secret@lab.example",
    "https://lab.example?token=secret", "https://magnitude.dev", "https://api.magnitude.dev", "https://github.com",
    "https://release-assets.githubusercontent.com",
  ])("rejects invalid or production origins: %s", origin => {
    expect(Schema.is(ArtifactDelivery)({ _tag: "PrivateAcceptance", origin })).toBe(false)
  })
  it("does not admit private URLs through the default production policy", () => {
    expect(acceptsArtifactUrl({ _tag: "Github" }, asset)).toBe(false)
  })
})
