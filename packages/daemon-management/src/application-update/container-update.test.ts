import { BunContext } from "@effect/platform-bun"
import { generateKeyPairSync } from "node:crypto"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Effect, Option, Schema, Stream } from "effect"
import { describe, expect, it } from "vitest"
import type { UpdateOutcome } from "@magnitudedev/release/hosted-update"
import { UpdateManifest, signUpdateManifest } from "../../../release/src/hosted-update/manifest"
import { PreparedUpdateStore } from "../desktop-native/prepared-update"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"
import { containerImageVersions, containerUpdateRefusal, makeContainerApplicationUpdate, recordContainerUpgrade } from "./container-update"

const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-linux-x64-deb", target: { os: "linux", arch: "x64", package: "deb" }, filename: "magnitude.deb", bytes: 100, sha256: "a".repeat(64),
} })
const candidate = (await Effect.runPromise(signUpdateManifest(manifest, generateKeyPairSync("ed25519").privateKey))).release

const source = (check: ApplicationUpdateSource["check"]) => ApplicationUpdateSource.of({ check,
  download: () => Effect.die("A container never downloads an update"), stage: () => Effect.die("A container never stages an update") })
const settled = <A>(changes: Stream.Stream<A & { readonly check: { readonly _tag: string } }>) =>
  changes.pipe(Stream.filter(state => state.check._tag !== "Checking" && state.check._tag !== "Idle"), Stream.runHead)

describe("container updates", () => {
  it("announces a newer release as an image and keeps checking, but never downloads or installs it", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const offers = [Option.some(candidate), Option.none()]
    const reasons: string[] = []
    const updates = yield* makeContainerApplicationUpdate.pipe(Effect.provideService(ApplicationUpdateSource,
      source(reason => Effect.sync(() => { reasons.push(reason); return offers.shift()! }))))
    yield* updates.check("launch")
    yield* settled(updates.changes)
    expect((yield* updates.state).transfer).toEqual({ _tag: "ImageAvailable", version: "2.0.0" })
    for (const refused of [updates.download, updates.discard, updates.requireReady, updates.setAutoDownload(true)]) {
      expect((yield* Effect.flip(refused)).message).toBe(containerUpdateRefusal)
    }
    expect(containerUpdateRefusal).toContain("docker pull magnitudedev/magnitude")
    yield* updates.check("scheduled")
    yield* updates.changes.pipe(Stream.filter(state => state.transfer._tag === "Idle"), Stream.runHead)
    expect(reasons).toEqual(["launch", "scheduled"])
  }))))
  it("keeps an announced image when a later check fails", () => Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    let calls = 0
    const updates = yield* makeContainerApplicationUpdate.pipe(Effect.provideService(ApplicationUpdateSource,
      source(() => calls++ === 0 ? Effect.succeed(Option.some(candidate)) : new ApplicationUpdateFailed({ message: "Could not check for application updates." }))))
    yield* updates.check("launch")
    yield* settled(updates.changes)
    yield* updates.check("manual")
    yield* updates.changes.pipe(Stream.filter(state => state.check._tag === "Failed"), Stream.runHead)
    expect((yield* updates.state).transfer).toEqual({ _tag: "ImageAvailable", version: "2.0.0" })
  }))))
  it("names each announced version once while it stays announced", async () => {
    const state = (transfer: { readonly _tag: "Idle" } | { readonly _tag: "ImageAvailable"; readonly version: string }) =>
      ({ transfer, check: { _tag: "Idle" }, preference: { _tag: "Unavailable", message: "" } }) as const
    const versions = await Effect.runPromise(Stream.runCollect(containerImageVersions(Stream.make(state({ _tag: "Idle" }),
      state({ _tag: "ImageAvailable", version: "2.0.0" }), state({ _tag: "ImageAvailable", version: "2.0.0" }), state({ _tag: "ImageAvailable", version: "2.1.0" })))))
    expect([...versions]).toEqual(["2.0.0", "2.1.0"])
  })
  it("reports an upgrade only when a newer version starts on the data directory", async () => {
    const root = await mkdtemp(join(tmpdir(), "magnitude-container-upgrade-"))
    const recorded: UpdateOutcome[] = []
    const store = { recordOutcome: (outcome: UpdateOutcome) => Effect.sync(() => { recorded.push(outcome) }) } as unknown as PreparedUpdateStore
    const start = (version: string) => Effect.runPromise(recordContainerUpgrade(root, version).pipe(
      Effect.provideService(PreparedUpdateStore, store), Effect.provide(BunContext.layer)))
    try {
      await start("1.0.0")
      await start("1.0.0")
      expect(recorded).toEqual([])
      await start("1.1.0")
      expect(recorded).toEqual([{ outcome: "applied", version: "1.1.0", from: Option.some("1.0.0"), reason: Option.none() }])
      await start("1.0.5")
      await start("1.0.5")
      expect(recorded).toHaveLength(1)
      await start("1.2.0-beta.1")
      expect(recorded[1]).toEqual({ outcome: "applied", version: "1.2.0-beta.1", from: Option.some("1.0.5"), reason: Option.none() })
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
