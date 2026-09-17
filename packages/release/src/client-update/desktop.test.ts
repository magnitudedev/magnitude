import { FetchHttpClient } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Option } from "effect"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { describe, expect, it } from "vitest"
import { findMacDesktopUpdate } from "./desktop"

const host = "darwin-arm64" as const
const archive = { id: `desktop-update-${host}`, kind: "desktop", host, filename: `magnitude-desktop-${host}.zip`, bytes: 1234, sha256: "b".repeat(64) }
const manifest = (version: string, artifact: object) => ({
  schemaVersion: 2, version, acnRevision: 1, rpc: { version: 1, fingerprint: "a".repeat(64) },
  plugins: [{ host: "pi", name: "@magnitudedev/pi-extension", version: "0.0.1", rpcVersion: 1, contentFingerprint: "a".repeat(64), filename: "pi.tgz", integrity: "sha512-dGVzdA==" }],
  tag: `@magnitudedev/cli@${version}`, sourceCommit: "a".repeat(40), artifacts: [artifact],
})

describe("Mac desktop update selection", () => {
  it("skips a newer CLI-only release and returns the exact desktop archive", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-desktop-update-"))
    const server = Bun.serve({ port: 0, fetch(request) {
      const url = decodeURIComponent(new URL(request.url).pathname)
      if (url === "/registry") return Response.json({ latest: "2.0.0", beta: "1.5.0-beta.1" })
      const version = /cli@([^/]+)/.exec(url)?.[1]
      return Response.json(manifest(version!, version === "2.0.0" ? { ...archive, id: `cli-${host}`, kind: "cli", filename: "magnitude-cli.tar.gz" } : archive))
    } })
    try {
      const result = await Effect.runPromise(findMacDesktopUpdate({ currentVersion: "1.0.0-beta.1", host,
        registryUrl: `${server.url}registry`, releaseBaseUrl: `${server.url}releases`, cacheDirectory: root,
      }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
      expect(Option.getOrThrow(result)).toMatchObject({ version: "1.5.0-beta.1", artifact: { id: archive.id, filename: archive.filename, bytes: archive.bytes, sha256: archive.sha256 } })
    } finally { server.stop(true); await rm(root, { recursive: true, force: true }) }
  })

  it.each([
    { ...archive, host: "darwin-x64" },
    { ...archive, filename: "magnitude-desktop-darwin-arm64.dmg" },
    { ...archive, id: "desktop-darwin-arm64" },
  ])("rejects an archive with wrong host, format or identity: $id / $filename / $host", async artifact => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-desktop-update-"))
    const server = Bun.serve({ port: 0, fetch(request) {
      return Response.json(new URL(request.url).pathname === "/registry" ? { latest: "2.0.0" } : manifest("2.0.0", artifact))
    } })
    try {
      const result = await Effect.runPromise(findMacDesktopUpdate({ currentVersion: "1.0.0", host,
        registryUrl: `${server.url}registry`, releaseBaseUrl: `${server.url}releases`, cacheDirectory: root,
      }).pipe(Effect.provide([BunContext.layer, FetchHttpClient.layer])))
      expect(Option.isNone(result)).toBe(true)
    } finally { server.stop(true); await rm(root, { recursive: true, force: true }) }
  })
})
