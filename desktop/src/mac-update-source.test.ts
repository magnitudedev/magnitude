import { createHash } from "node:crypto"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option, Schema, Stream } from "effect"
import { DesktopUpdateCandidate } from "@magnitudedev/release"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateSource, makeApplicationUpdate } from "./application-update"
import { NativeMacUpdate } from "./mac-update-stage"
import { macUpdateSource } from "./mac-update-source"
import { ApplicationUpdateHandoff } from "./update-handoff"

describe("Mac update acquisition and native handoff", () => {
  it.each([false, true])("verifies downloaded bytes before native staging (corrupt=%s)", async corrupt => {
    const bytes = Buffer.from("verified ZIP fixture")
    const candidate = Schema.decodeUnknownSync(DesktopUpdateCandidate)({ version: "2.0.0", artifact: {
      id: "desktop-update-darwin-arm64", kind: "desktop", host: "darwin-arm64",
      filename: "magnitude-desktop-darwin-arm64.zip", bytes: bytes.length,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    } })
    const root = await mkdtemp(join(tmpdir(), "mac-update-source-"))
    const server = Bun.serve({ port: 0, fetch: () => new Response(corrupt ? Buffer.alloc(bytes.length, 0) : bytes) })
    let staged = false
    let recorded = false
    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const source = yield* macUpdateSource({ currentVersion: "1.0.0", host: "darwin-arm64", releaseBaseUrl: String(server.url), cacheDirectory: root }).pipe(
          Effect.provideService(ApplicationUpdateHandoff, { record: version => Effect.sync(() => { expect(version).toBe(candidate.version); recorded = true }), inspect: () => Effect.succeed({ _tag: "Continue" as const }) }),
          Effect.provideService(NativeMacUpdate, {
            stage: feed => Effect.promise(async () => {
              staged = true
              expect(recorded).toBe(true)
              const manifest = await (await fetch(feed.url, { headers: { Authorization: feed.authorization } })).json() as { url: string }
              expect(Buffer.from(await (await fetch(manifest.url)).arrayBuffer())).toEqual(bytes)
            }),
          }),
        )
        const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, { ...source, check: Effect.succeed(Option.some(candidate)) }))
        yield* owner.check
        yield* owner.changes.pipe(Stream.filter(state => state._tag === "Available"), Stream.take(1), Stream.runDrain)
        yield* owner.download
        yield* owner.changes.pipe(Stream.filter(state => state._tag === (corrupt ? "Failed" : "Ready")), Stream.take(1), Stream.runDrain)
        yield* owner.close
      })).pipe(Effect.timeout("5 seconds")))
      expect(staged).toBe(!corrupt)
      expect(recorded).toBe(!corrupt)
      expect(await readdir(root)).toEqual([])
    } finally { server.stop(true); await rm(root, { recursive: true, force: true }) }
  })
})
