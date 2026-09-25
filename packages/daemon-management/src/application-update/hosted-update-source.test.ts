import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Either, Option, Ref, Schema } from "effect"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { describe, expect, it, vi } from "vitest"
import { UpdateClientMetadata, UpdateRelease, type UpdateCheck, type UpdateOutcome } from "@magnitudedev/release/hosted-update"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { hostedUpdateSource } from "./hosted-update-source"

const checks: UpdateCheck[] = []
let serverAvailable = true
vi.mock("@magnitudedev/release/hosted-update", async importOriginal => {
  const actual = await importOriginal<typeof import("@magnitudedev/release/hosted-update")>()
  return {
    ...actual,
    checkHostedUpdate: (_options: unknown, check: UpdateCheck) => Effect.suspend(() => {
      checks.push(check)
      return serverAvailable ? Effect.succeed(Option.none()) : Effect.fail(new actual.HostedUpdateCheckFailed({ reason: "network" }))
    }),
    resolveHostedDownload: () => Effect.succeed(new URL("https://example.com/installer")),
    downloadUpdateArtifact: (options: Parameters<typeof actual.downloadUpdateArtifact>[0]) => Effect.gen(function* () {
      const fs = yield* FileSystem.FileSystem
      yield* fs.writeFileString(options.destination, "verified transfer fixture")
      return { destination: options.destination }
    }),
  }
})

const store = (outcome: Option.Option<UpdateOutcome>, reported: Ref.Ref<number>): PreparedUpdateStore => ({
  outcome: Effect.succeed(outcome), markOutcomeReported: Ref.update(reported, n => n + 1), recordOutcome: () => Effect.void,
  read: Effect.succeed(Option.none()), prepare: () => Effect.void, verify: () => Effect.die("unused"), recordAttempt: () => Effect.void,
  recordFailure: () => Effect.void, discard: Effect.void, removeAbandonedTransfers: Effect.void,
})
const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "11", arch: "x64", package: "windows-exe" })
const options = { origin: "https://example.com", metadata, dataDirectory: "/unused", userAgent: "fixture", sign: () => Effect.succeed("unused"), trustedPublishers: new Map() }

describe("hosted update checks", () => {
  it("sends a pending outcome with the check and marks it reported only after the server answers", async () => {
    checks.length = 0
    const outcome: UpdateOutcome = { outcome: "failed", version: "1.0.1", reason: Option.some("install") }
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const reported = yield* Ref.make(0)
      const source = yield* hostedUpdateSource(options, () => Effect.void).pipe(Effect.provideService(PreparedUpdateStore, store(Option.some(outcome), reported)))
      serverAvailable = false
      expect(Either.isLeft(yield* Effect.either(source.check("launch")))).toBe(true)
      expect(yield* Ref.get(reported)).toBe(0)
      serverAvailable = true
      expect(Option.isNone(yield* source.check("scheduled"))).toBe(true)
      expect(yield* Ref.get(reported)).toBe(1)
      const silent = yield* hostedUpdateSource(options, () => Effect.void).pipe(Effect.provideService(PreparedUpdateStore, store(Option.none(), reported)))
      yield* silent.check("manual")
      expect(yield* Ref.get(reported)).toBe(1)
    })).pipe(Effect.provide(BunContext.layer)))
    expect(checks).toEqual([{ reason: "launch", outcome: Option.some(outcome) }, { reason: "scheduled", outcome: Option.some(outcome) }, { reason: "manual", outcome: Option.none() }])
  })
})

describe("hosted update transfer storage", () => {
  it("never pre-creates the private prepared directory and removes its scoped transfer", async () => {
    const root = await mkdtemp(join(tmpdir(), "hosted-update-transfer-"))
    const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "11", arch: "x64", package: "windows-exe" })
    const release = Schema.decodeUnknownSync(UpdateRelease)({ version: "2.0.0", bytes: 25, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" })

    try {
      await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
        const source = yield* hostedUpdateSource({ origin: "https://example.com", metadata, dataDirectory: root,
          userAgent: "fixture", sign: () => Effect.succeed("unused"), trustedPublishers: new Map() }, () => Effect.void).pipe(
            Effect.provideService(PreparedUpdateStore, store(Option.none(), yield* Ref.make(0))))
        const fs = yield* FileSystem.FileSystem
        const archive = yield* source.download(release, () => Effect.void)
        expect(dirname(dirname(archive))).toBe(join(root, "update-downloads"))
        expect(yield* fs.exists(join(root, "updates"))).toBe(false)
        expect(yield* fs.readFileString(archive)).toBe("verified transfer fixture")
      })).pipe(Effect.provide(BunContext.layer)))
      expect(await readdir(join(root, "update-downloads"))).toEqual([])
      expect(await readdir(root)).toEqual(["update-downloads"])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
