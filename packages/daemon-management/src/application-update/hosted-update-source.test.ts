import { FileSystem } from "@effect/platform"
import { BunContext } from "@effect/platform-bun"
import { Effect, Either, Option, Ref, Schema } from "effect"
import { mkdtemp, readdir, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { describe, expect, it, vi } from "vitest"
import { UpdateClientMetadata, UpdateRelease, type UpdateCheck, type UpdateHardware, type UpdateOutcome } from "@magnitudedev/release/hosted-update"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { hostedUpdateSource } from "./hosted-update-source"
import { UpdateHardwareSource } from "./update-hardware"

const checks: UpdateCheck[] = []
let serverAvailable = true
let downloadFailure: "integrity" | "stream" | undefined
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
      if (downloadFailure !== undefined) return yield* Effect.fail({ _tag: "ArtifactDownloadError", phase: downloadFailure, message: "fixture" })
      const fs = yield* FileSystem.FileSystem
      yield* fs.writeFileString(options.destination, "verified transfer fixture")
      return { destination: options.destination }
    }),
  }
})

const store = (outcome: Option.Option<UpdateOutcome>, reported: Ref.Ref<number>): PreparedUpdateStore => ({
  outcome: Effect.succeed(outcome), markOutcomeReported: Ref.update(reported, n => n + 1), recordOutcome: () => Effect.void, hold: Effect.succeed(Option.none()), recordHold: () => Effect.void, releaseHold: Effect.void,
  read: Effect.succeed(Option.none()), prepare: () => Effect.void, verify: () => Effect.die("unused"), recordAttempt: () => Effect.void,
  recordFailure: () => Effect.void, complete: () => Effect.void, discard: Effect.void, removeAbandonedTransfers: Effect.void,
})
const hardwareSource = (hardware: Option.Option<UpdateHardware>) => ({ current: Effect.succeed(hardware) })
const cuda: UpdateHardware = { backend: Option.some("cuda"), gpu_name: Option.some("NVIDIA A10-8Q"), gpu_kind: Option.some("discrete"), gpu_memory_gb: Option.some(8), gpu_count: Option.some(1),
  gpu_unused_reason: Option.none(), cpu_model: Option.some("AMD EPYC 7V13"), cpu_cores: Option.some(12), memory_gb: Option.some(110) }
const metadata = Schema.decodeUnknownSync(UpdateClientMetadata)({ version: "1.0.0", os: "windows", os_version: "11", arch: "x64", package: "windows-exe" })
const options = { origin: "https://example.com", metadata, dataDirectory: "/unused", userAgent: "fixture", sign: () => Effect.succeed("unused"), trustedPublishers: new Map() }

describe("hosted update checks", () => {
  it("sends a pending outcome with the check and marks it reported only after the server answers", async () => {
    checks.length = 0
    const outcome: UpdateOutcome = { outcome: "failed", version: "1.0.1", from: Option.some("1.0.0"), reason: Option.some("install") }
    await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
      const reported = yield* Ref.make(0)
      const source = yield* hostedUpdateSource(options, () => Effect.void).pipe(Effect.provideService(PreparedUpdateStore, store(Option.some(outcome), reported)), Effect.provideService(UpdateHardwareSource, hardwareSource(Option.some(cuda))))
      serverAvailable = false
      expect(Either.isLeft(yield* Effect.either(source.check("launch")))).toBe(true)
      expect(yield* Ref.get(reported)).toBe(0)
      serverAvailable = true
      expect(Option.isNone(yield* source.check("scheduled"))).toBe(true)
      expect(yield* Ref.get(reported)).toBe(1)
      const silent = yield* hostedUpdateSource(options, () => Effect.void).pipe(Effect.provideService(PreparedUpdateStore, store(Option.none(), reported)), Effect.provideService(UpdateHardwareSource, hardwareSource(Option.none())))
      yield* silent.check("manual")
      expect(yield* Ref.get(reported)).toBe(1)
    })).pipe(Effect.provide(BunContext.layer)))
    expect(checks).toEqual([{ reason: "launch", outcome: Option.some(outcome), hardware: Option.some(cuda) }, { reason: "scheduled", outcome: Option.some(outcome), hardware: Option.some(cuda) }, { reason: "manual", outcome: Option.none(), hardware: Option.none() }])
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
            Effect.provideService(PreparedUpdateStore, store(Option.none(), yield* Ref.make(0))), Effect.provideService(UpdateHardwareSource, hardwareSource(Option.none())))
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

describe("hosted update transfer failures", () => {
  it("says a download that fails verification was refused, and any other failure that the download failed", async () => {
    const release = Schema.decodeUnknownSync(UpdateRelease)({ version: "2.0.0", bytes: 25, sha256: "a".repeat(64), signature: "A".repeat(86) + "==" })
    const root = await mkdtemp(join(tmpdir(), "hosted-update-failure-"))
    const messages = async (phase: "integrity" | "stream") => {
      downloadFailure = phase
      try {
        return await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
          const source = yield* hostedUpdateSource({ ...options, dataDirectory: root }, () => Effect.void).pipe(
            Effect.provideService(PreparedUpdateStore, store(Option.none(), yield* Ref.make(0))), Effect.provideService(UpdateHardwareSource, hardwareSource(Option.none())))
          return yield* source.download(release, () => Effect.void).pipe(Effect.flip, Effect.map(error => error.message))
        })).pipe(Effect.provide(BunContext.layer)))
      } finally { downloadFailure = undefined }
    }
    try {
      expect(await messages("integrity")).toBe("The download of Magnitude 2.0.0 did not match its signed size and checksum, so it was not installed.")
      expect(await messages("stream")).toBe("Could not download Magnitude 2.0.0.")
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
