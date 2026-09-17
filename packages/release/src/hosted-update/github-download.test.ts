import { FetchHttpClient, HttpClientRequest } from "@effect/platform"
import { Effect, Either } from "effect"
import { describe, expect, it } from "vitest"
import { githubDownloadClient } from "./github-download"
import { isGithubReleaseAssetUrl } from "./github-artifact"

const asset = "https://github.com/magnitudedev/magnitude/releases/download/test/app.exe"
const delivery = "https://release-assets.githubusercontent.com/github-production-release-asset/123/abc?sig=test"
const run = (fetcher: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>) => Effect.runPromise(Effect.gen(function* () {
  const client = yield* githubDownloadClient
  return yield* client.execute(HttpClientRequest.get(asset, { headers: {
    authorization: "installation-secret", cookie: "private", "x-api-key": "secret", range: "bytes=0-0", "if-range": '"etag"',
  } }))
}).pipe(Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(fetcher, { preconnect: () => {} })), Effect.either))

describe("GitHub installer delivery", () => {
  it("follows delivery redirects manually without leaking installation credentials", async () => {
    const calls: string[] = []
    const result = await run(async (input, init) => {
      const request = new Request(input, init)
      calls.push(request.url)
      expect(init?.redirect).toBe("manual")
      expect(init?.credentials).toBe("omit")
      for (const name of ["authorization", "cookie", "x-api-key"]) expect(request.headers.has(name)).toBe(false)
      expect(request.headers.get("range")).toBe("bytes=0-0")
      expect(request.headers.get("if-range")).toBe('"etag"')
      return calls.length === 1 ? new Response(null, { status: 302, headers: { location: delivery } }) : new Response("x", { status: 206 })
    })
    expect(Either.isRight(result)).toBe(true)
    expect(calls).toEqual([asset, delivery])
  })
  it.each([
    "https://evil.example/file", "http://release-assets.githubusercontent.com/github-production-release-asset/123/abc",
    "https://github.com/other/repo/releases/download/test/app.exe", "https://github.com/magnitudedev/magnitude/issues/1",
    "https://user:secret@release-assets.githubusercontent.com/github-production-release-asset/123/abc",
    "https://release-assets.githubusercontent.com:8443/github-production-release-asset/123/abc",
    "https://release-assets.githubusercontent.com/other/abc", "https://release-assets.githubusercontent.com.evil.example/github-production-release-asset/123/abc",
  ])("rejects an untrusted redirect before sending a request: %s", async location => {
    let calls = 0
    expect(Either.isLeft(await run(async () => { calls++; return new Response(null, { status: 302, headers: { location } }) }))).toBe(true)
    expect(calls).toBe(1)
  })
  it("bounds redirect loops", async () => {
    let calls = 0
    expect(Either.isLeft(await run(async () => { calls++; return new Response(null, { status: 302, headers: { location: delivery } }) }))).toBe(true)
    expect(calls).toBe(5)
  })
  it("admits only exact release URLs in the Magnitude repository", () => {
    expect(isGithubReleaseAssetUrl(asset)).toBe(true)
    for (const url of [asset + "?token=secret", asset + "#fragment", asset.replace("github.com", "github.com.evil.example"), asset.replace("test/app.exe", "test%2Fapp.exe"), asset.replace("https:", "http:")]) {
      expect(isGithubReleaseAssetUrl(url)).toBe(false)
    }
  })
})
