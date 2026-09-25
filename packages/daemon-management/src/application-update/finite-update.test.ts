import { Deferred, Effect, Exit, Fiber, Option, Ref, Scope } from "effect"
import { createHash, generateKeyPairSync } from "node:crypto"
import { describe, expect, it } from "vitest"
import { signUpdateRelease } from "../../../release/src/hosted-update/release"
import { PreparedUpdateStore, PreparedUpdateFailed, type PreparedUpdate } from "../desktop-native/prepared-update"
import { UpdatePreferences } from "../desktop-native/update-preferences"
import { ApplicationUpdateFailed, ApplicationUpdateSource } from "./application-update"
import { readPreparedUpdateState, runFiniteUpdatePreparation } from "./finite-update"

const key = generateKeyPairSync("ed25519")
const release = await Effect.runPromise(signUpdateRelease({ version: "0.1.6", bytes: 1,
  sha256: createHash("sha256").update("x").digest("hex") }, { os: "darwin", arch: "arm64", package: "mac-zip" }, key.privateKey))
const fixture = Effect.gen(function* () {
  const pending = yield* Ref.make<Option.Option<PreparedUpdate>>(Option.none())
  const events: string[] = []
  const event = (name: string) => Effect.sync(() => { events.push(name) })
  const store = PreparedUpdateStore.of({
    read: event("read").pipe(Effect.zipRight(Ref.get(pending))),
    prepare: (_archive, candidate) => event("publish").pipe(Effect.zipRight(Ref.set(pending, Option.some({ release: candidate, installation: { _tag: "Unattempted" } })))),
    discard: event("discard").pipe(Effect.zipRight(Ref.set(pending, Option.none()))),
    removeAbandonedTransfers: event("scratch-cleanup"),
    outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void,
    verify: () => Effect.die("Preparation cannot install"), recordAttempt: () => Effect.die("Preparation cannot attempt installation"),
    recordFailure: () => Effect.die("Preparation cannot record installation failure"),
  })
  const source = ApplicationUpdateSource.of({
    check: () => event("check").pipe(Effect.as(Option.some(release))),
    download: () => Effect.acquireRelease(event("download").pipe(Effect.as("archive")), () => event("retire-transfer")),
    stage: archive => store.prepare(archive, release).pipe(Effect.mapError(error => new ApplicationUpdateFailed({ message: error.message }))),
  })
  const provide = <A, E, R>(effect: Effect.Effect<A, E, R>) => effect.pipe(Effect.provideService(PreparedUpdateStore, store),
    Effect.provideService(ApplicationUpdateSource, source), Effect.provideService(UpdatePreferences, {
      read: Effect.succeed(true), write: () => Effect.die("Finite commands must not change preferences"),
    }))
  return { pending, events, event, store, source, provide }
})
const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope>) => Effect.runPromise(effect.pipe(Effect.scoped, Effect.timeout("3 seconds")))

describe("finite application update preparation", () => {
  it("reads saved state without checking or mutating", () => run(Effect.gen(function* () {
    const { events, provide } = yield* fixture
    expect((yield* provide(readPreparedUpdateState)).transfer._tag).toBe("Idle")
    expect(events).toEqual(["read"])
  })))
  it("checks without starting an automatic download even when the preference is enabled", () => run(Effect.gen(function* () {
    const { events, provide } = yield* fixture
    expect((yield* provide(runFiniteUpdatePreparation("check"))).transfer).toEqual({ _tag: "Available", version: "0.1.6", bytes: 1 })
    expect(events).toEqual(["read", "check"])
  })))
  it("returns Ready only after durable publication and transfer retirement", () => run(Effect.gen(function* () {
    const { events, provide } = yield* fixture
    expect((yield* provide(runFiniteUpdatePreparation("download"))).transfer).toEqual({ _tag: "Ready", version: "0.1.6" })
    expect(events).toEqual(["read", "check", "scratch-cleanup", "download", "publish", "read", "retire-transfer"])
  })))
  it("waits for publication and cleans up when cancelled", () => run(Effect.gen(function* () {
    const { events, provide, source, event, pending } = yield* fixture
    const entered = yield* Deferred.make<void>()
    const worker = yield* provide(runFiniteUpdatePreparation("download").pipe(Effect.provideService(ApplicationUpdateSource, {
      ...source, stage: () => event("staging").pipe(Effect.zipRight(Deferred.succeed(entered, undefined)), Effect.zipRight(Effect.never)),
    }))).pipe(Effect.forkScoped)
    yield* Deferred.await(entered)
    expect(Exit.isInterrupted(yield* Fiber.interrupt(worker))).toBe(true)
    expect(events.at(-1)).toBe("retire-transfer")
    expect(Option.isNone(yield* Ref.get(pending))).toBe(true)
  })))
  it("does not claim readiness if staging returns without publishing", () => run(Effect.gen(function* () {
    const { events, provide, source } = yield* fixture
    expect(yield* provide(runFiniteUpdatePreparation("download").pipe(Effect.provideService(ApplicationUpdateSource, {
      ...source, stage: () => Effect.void,
    }))).pipe(Effect.isFailure)).toBe(true)
    expect(events.at(-1)).toBe("retire-transfer")
  })))
  it.each(["Unattempted", "Attempted", "Failed"] as const)("preserves an existing %s installer without redownloading", tag => run(Effect.gen(function* () {
    const { pending, events, provide } = yield* fixture
    yield* Ref.set(pending, Option.some({ release, installation: tag === "Failed" ? { _tag: tag, reason: "Install failed" } : { _tag: tag } }))
    expect((yield* provide(runFiniteUpdatePreparation("download"))).transfer._tag).toBe(tag === "Unattempted" ? "Ready" : "InstallationFailed")
    expect(events).toEqual(["read"])
  })))
  it("keeps retained state when discard fails", () => run(Effect.gen(function* () {
    const { pending, provide, store } = yield* fixture
    yield* Ref.set(pending, Option.some({ release, installation: { _tag: "Unattempted" } }))
    expect(yield* provide(runFiniteUpdatePreparation("discard").pipe(Effect.provideService(PreparedUpdateStore, {
      ...store, discard: new PreparedUpdateFailed({ message: "Cleanup failed" }),
    }))).pipe(Effect.isFailure)).toBe(true)
    expect(Option.isSome(yield* Ref.get(pending))).toBe(true)
  })))
})
