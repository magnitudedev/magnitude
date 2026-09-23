import { FileSystem } from "@effect/platform"
import { NodeContext } from "@effect/platform-node"
import { Effect, Schema } from "effect"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { describe, expect, it, vi } from "vitest"
import { UpdateClientMetadata, UpdateRelease } from "@magnitudedev/release/hosted-update"
import { hostedUpdateSource } from "./hosted-update-source"

vi.mock("@magnitudedev/release/hosted-update", async importOriginal => {
  const actual = await importOriginal<typeof import("@magnitudedev/release/hosted-update")>()
  return {
    ...actual,
    resolveHostedDownload: () => Effect.succeed(new URL("https://example.com/installer")),
    downloadUpdateArtifact: (options: Parameters<typeof actual.downloadUpdateArtifact>[0]) => Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      yield* fs.writeFileString(options.destination, "verified transfer fixture")
      return { destination: options.destination }
    }),
  }
})

describe("hosted update transfer storage", () => {
  it("never pre-creates the private prepared directory and removes its scoped transfer", async () => {
    const root = await mkdtemp(join(tmpdir(), "hosted-update-transfer-"))
    const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "11", arch: "x64", package: "windows-exe" })
    const release = Schema.decodeUnknownSync(UpdateRelease)({ version: "2.0.0", bytes: 25, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" })
    const source = hostedUpdateSource({ origin: "https://example.com", metadata, dataDirectory: root,
      userAgent: "fixture", sign: () => Effect.succeed("unused"), trustedPublishers: new Map() }, () => Effect.void)
    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const fs = yield* FileSystem.FileSystem
        const archive = yield* source.download(release, () => Effect.void)
        expect(dirname(dirname(archive))).toBe(join(root, "update-downloads"))
        expect(yield* fs.exists(join(root, "updates"))).toBe(false)
        expect(yield* fs.readFileString(archive)).toBe("verified transfer fixture")
      })).pipe(Effect.provide(NodeContext.layer)))
      expect(await readdir(join(root, "update-downloads"))).toEqual([])
      expect(await readdir(root)).toEqual(["update-downloads"])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
