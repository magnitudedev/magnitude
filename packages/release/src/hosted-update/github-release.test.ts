import { FetchHttpClient } from "@effect/platform"
import { Effect, Option, Schema } from "effect"
import { describe, expect, it } from "vitest"
import { verifyGithubRelease } from "./github-release"
import { UpdateManifest } from "./manifest"

const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", tag: "@magnitudedev/cli@2.0.0", commit: "a".repeat(40),
  artifact: { id: "mac", target: { os: "darwin", arch: "arm64", package: "dmg" }, filename: "Magnitude.dmg", bytes: 100, sha256: "b".repeat(64) } })
const asset = { name: "Magnitude.dmg", size: 100, digest: `sha256:${"b".repeat(64)}`, state: "uploaded" }
const release = { tag_name: manifest.tag, draft: false, assets: [asset] }
const run = (value: unknown = release, commit = manifest.commit, status = 200) => {
  const calls: string[] = []
  return Effect.runPromise(verifyGithubRelease([manifest], Option.some("ci-test-token")).pipe(
    Effect.provide(FetchHttpClient.layer), Effect.provideService(FetchHttpClient.Fetch, Object.assign(async (input: RequestInfo | URL, init?: RequestInit) => {
      const request = new Request(input, init); calls.push(request.url)
      expect(new URL(request.url).origin).toBe("https://api.github.com")
      expect(request.headers.get("authorization")).toBe("Bearer ci-test-token")
      return Response.json(request.url.includes('/commits/') ? { sha: commit } : value, { status })
    }, { preconnect: () => {} })), Effect.either,
  )).then(result => ({ result, calls }))
}
describe("GitHub release metadata registration", () => {
  it("validates the exact public source and accepted asset without downloading binary bytes", async () => {
    const { result, calls } = await run()
    expect(result._tag).toBe("Right")
    expect(calls).toHaveLength(2)
    expect(calls.every(url => !url.includes('/releases/download/'))).toBe(true)
  })
  it.each([
    { ...release, draft: true }, { ...release, tag_name: 'other' }, { ...release, assets: [] },
    { ...release, assets: [asset, asset] }, { ...release, assets: [{ ...asset, size: 99 }] },
    { ...release, assets: [{ ...asset, digest: null }] }, { ...release, assets: [{ ...asset, digest: `sha256:${'c'.repeat(64)}` }] },
    { ...release, assets: [{ ...asset, state: 'new' }] },
  ])("rejects unaccepted release metadata", async value => expect((await run(value)).result._tag).toBe("Left"))
  it("resolves the tag to the exact commit", async () => expect((await run(release, 'c'.repeat(40))).result._tag).toBe("Left"))
  it.each([403, 404, 429, 503])("refuses unavailable GitHub metadata (%s)", async status => expect((await run(release, manifest.commit, status)).result._tag).toBe("Left"))
})
