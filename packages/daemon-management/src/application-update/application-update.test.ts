import { generateKeyPairSync } from 'node:crypto'
import { Deferred, Effect, Option, Ref, Schema, Scope, Stream } from "effect"
import { UpdateManifest, PublisherKeyId, signUpdateManifest } from "../../../release/src/hosted-update/manifest"
import { PreparedUpdateStore, PreparedUpdateFailed, UpdatePreferences, UpdatePreferencesFailed, type PreparedUpdate, type UnattendedHold } from "@magnitudedev/daemon-management/desktop-native"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate, type ApplicationUpdate } from "./application-update"

const manifest = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, tag: "@magnitudedev/cli@2.0.0", version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" },
  filename: "magnitude-desktop-darwin-arm64.zip", bytes: 100, sha256: "a".repeat(64),
} })
const candidate = (await Effect.runPromise(signUpdateManifest(manifest, generateKeyPairSync("ed25519").privateKey))).release
const waitFor = (owner: ApplicationUpdate, tag: string) => owner.changes.pipe(Stream.filter(state => state.transfer._tag === tag), Stream.take(1), Stream.runDrain)
const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope | UpdatePreferences | PreparedUpdateStore>) => Effect.runPromise(effect.pipe(Effect.provideService(PreparedUpdateStore, { read: Effect.succeed(Option.none()), prepare: () => Effect.void, verify: () => Effect.die("Unexpected verification"), recordAttempt: () => Effect.void, recordFailure: () => Effect.void, complete: () => Effect.void, discard: Effect.void, removeAbandonedTransfers: Effect.void, outcome: Effect.succeed(Option.none()), recordOutcome: () => Effect.void, markOutcomeReported: Effect.void, hold: Effect.succeed(Option.none()), recordHold: () => Effect.void, releaseHold: Effect.void }), Effect.provideService(UpdatePreferences, { read: Effect.succeed(false), write: () => Effect.void }), Effect.scoped, Effect.timeout("3 seconds")))

describe("application-owned updates", () => {
  it.each([false, true])("discards a retained update only after durable cleanup succeeds (failure=%s)", fail => run(Effect.gen(function* () {
    const store = yield* PreparedUpdateStore
    const calls = yield* Ref.make(0)
    const owner = yield* makeApplicationUpdate(Option.some({ release: candidate, from: Option.none(), installation: { _tag: "Failed", kind: "verify", reason: "Invalid retained bytes" } })).pipe(
      Effect.provideService(PreparedUpdateStore, { ...store, discard: Ref.update(calls, n => n + 1).pipe(Effect.zipRight(fail ? new PreparedUpdateFailed({ message: "Cleanup failed" }) : Effect.void)) }),
      Effect.provideService(ApplicationUpdateSource, { check: () => Effect.succeed(Option.some(candidate)), download: () => Effect.die("Discard must not download"), stage: () => Effect.void }),
    )
    const outcome = yield* owner.discard.pipe(Effect.either)
    expect(outcome._tag).toBe(fail ? "Left" : "Right")
    expect(yield* Ref.get(calls)).toBe(1)
    expect((yield* owner.state).transfer._tag).toBe(fail ? "InstallationFailed" : "Idle")
    if (!fail) {
      expect((yield* owner.requireReady.pipe(Effect.either))._tag).toBe("Left")
      yield* owner.check("manual")
      yield* waitFor(owner, "Available")
    }
  })))
  it.each(["Unattempted", "Attempted", "Failed"] as const)("keeps a saved %s update across checks without downloading it again", tag => run(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate(Option.some({ release: candidate, from: Option.none(),
      installation: tag === "Failed" ? { _tag: tag, kind: "authorization", reason: "Authorization cancelled" } : { _tag: tag },
    })).pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(true), write: () => Effect.void }),
      Effect.provideService(ApplicationUpdateSource, {
        check: () => Effect.succeed(Option.some(candidate)),
        download: () => Effect.die("Retained installers must not be redownloaded"), stage: () => Effect.die("No new download"),
      }))
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
    expect((yield* owner.state).transfer).toEqual(tag === "Unattempted" ? { _tag: "Ready", version: candidate.version }
      : { _tag: "InstallationFailed", version: candidate.version, message: tag === "Failed" ? "Authorization cancelled" : "The update did not complete." })
    yield* owner.requireReady
  })))
  it("continues checking after a preference read failure and never claims an unsaved choice", () => run(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(UpdatePreferences, {
      read: new UpdatePreferencesFailed({ message: "Unreadable preferences" }),
      write: () => new UpdatePreferencesFailed({ message: "Read-only profile" }),
    }), Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)), download: () => Effect.die("Automatic download must remain paused"), stage: () => Effect.void,
    }))
    yield* owner.check("manual")
    yield* waitFor(owner, "Available")
    expect((yield* owner.setAutoDownload(true).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* owner.state).preference._tag).toBe("Unavailable")
    expect((yield* owner.state).transfer._tag).toBe("Available")
  })))
  it("keeps checking while an automatic download is active and after staging without replacing it", () => run(Effect.gen(function* () {
    const checks = yield* Ref.make(0)
    const finish = yield* Deferred.make<void>()
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(true), write: () => Effect.void }), Effect.provideService(ApplicationUpdateSource, {
      check: () => Ref.updateAndGet(checks, n => n + 1).pipe(Effect.map(n => n === 1 ? Option.some(candidate) : Option.none())),
      download: () => Deferred.await(finish).pipe(Effect.as("archive")), stage: () => Effect.void,
    }))
    yield* owner.check("manual")
    yield* waitFor(owner, "Downloading")
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
    expect(yield* Ref.get(checks)).toBe(2)
    expect((yield* owner.state).transfer._tag).toBe("Downloading")
    yield* Deferred.succeed(finish, undefined)
    yield* waitFor(owner, "Ready")
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
    expect(yield* Ref.get(checks)).toBe(3)
    yield* owner.requireReady
  })))

  it("waits for cancelled download cleanup before admitting a transfer after a quick off/on toggle", () => run(Effect.gen(function* () {
    const started = yield* Deferred.make<void>()
    const cleanup = yield* Deferred.make<void>()
    const releaseCleanup = yield* Deferred.make<void>()
    const downloads = yield* Ref.make(0)
    const staged = yield* Ref.make(false)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(true), write: () => Effect.void }), Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)),
      download: () => Effect.gen(function* () {
        const count = yield* Ref.updateAndGet(downloads, n => n + 1)
        if (count === 1) {
          yield* Effect.addFinalizer(() => Deferred.succeed(cleanup, undefined).pipe(Effect.zipRight(Deferred.await(releaseCleanup))))
          yield* Deferred.succeed(started, undefined)
          return yield* Effect.never
        }
        return "archive"
      }), stage: () => Ref.set(staged, true),
    }))
    yield* owner.check("manual")
    yield* Deferred.await(started)
    yield* owner.setAutoDownload(false)
    yield* Deferred.await(cleanup)
    yield* owner.setAutoDownload(true)
    expect((yield* owner.state).transfer._tag).toBe("Cancelling")
    expect((yield* owner.download.pipe(Effect.either))._tag).toBe("Left")
    expect(yield* Ref.get(downloads)).toBe(1)
    expect(yield* Ref.get(staged)).toBe(false)
    yield* Deferred.succeed(releaseCleanup, undefined)
    yield* waitFor(owner, "Ready")
    expect(yield* Ref.get(downloads)).toBe(2)
    expect(yield* Ref.get(staged)).toBe(true)
  })))

  it("turning automatic downloads off preserves an explicit user download", () => run(Effect.gen(function* () {
    const finish = yield* Deferred.make<void>()
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)), download: () => Deferred.await(finish).pipe(Effect.as("archive")), stage: () => Effect.void,
    }))
    yield* owner.check("manual")
    yield* waitFor(owner, "Available")
    yield* owner.download
    yield* owner.setAutoDownload(false)
    expect((yield* owner.state).transfer._tag).toBe("Downloading")
    yield* Deferred.succeed(finish, undefined)
    yield* waitFor(owner, "Ready")
  })))
  it("admits one worker, preserves the candidate, and retains it after observation ends", () => run(Effect.gen(function* () {
    const checks = yield* Ref.make(0)
    const finishCheck = yield* Deferred.make<void>()
    const finishStage = yield* Deferred.make<void>()
    const released = yield* Ref.make(false)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Ref.update(checks, n => n + 1).pipe(Effect.zipRight(Deferred.await(finishCheck)), Effect.as(Option.some(candidate))),
      download: (selected, report) => Effect.acquireRelease(Effect.gen(function* () {
        expect(selected).toEqual(candidate)
        yield* report(80)
        return "/verified/archive.zip"
      }), () => Ref.set(released, true)),
      stage: path => Effect.gen(function* () {
        expect(path).toBe("/verified/archive.zip")
        expect(yield* Ref.get(released)).toBe(false)
        yield* Deferred.await(finishStage)
      }),
    }))
    expect(yield* Ref.get(checks)).toBe(0)
    yield* owner.check("manual")
    expect((yield* owner.check("manual").pipe(Effect.either))._tag).toBe("Right")
    yield* Deferred.succeed(finishCheck, undefined)
    yield* waitFor(owner, "Available")
    expect(yield* Ref.get(checks)).toBe(1)
    yield* owner.download
    yield* waitFor(owner, "Staging")
    expect((yield* owner.requireReady.pipe(Effect.either))._tag).toBe("Left")
    // Downloading again while one is under way answers with that transfer instead of starting another.
    yield* owner.download
    expect((yield* owner.state).transfer._tag).toBe("Staging")
    // All observers above have ended. Their lifetime cannot end staging.
    yield* Deferred.succeed(finishStage, undefined)
    yield* waitFor(owner, "Ready")
    yield* owner.requireReady
    yield* owner.download
    expect((yield* owner.state).transfer._tag).toBe("Ready")
    yield* owner.close
    expect(yield* Ref.get(released)).toBe(true)
    expect((yield* owner.state).transfer._tag).toBe("Closed")
  })))

  it("Quit cancels an unfinished transfer, closes admission, and never stages it", () => run(Effect.gen(function* () {
    const started = yield* Deferred.make<void>()
    const cancelled = yield* Deferred.make<void>()
    const staged = yield* Ref.make(false)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)),
      download: () => Deferred.succeed(started, undefined).pipe(Effect.zipRight(Effect.never), Effect.ensuring(Deferred.succeed(cancelled, undefined))),
      stage: () => Ref.set(staged, true),
    }))
    yield* owner.check("manual")
    yield* waitFor(owner, "Available")
    yield* owner.download
    yield* Deferred.await(started)
    yield* owner.close
    yield* Deferred.await(cancelled)
    expect(yield* Ref.get(staged)).toBe(false)
    expect((yield* owner.check("manual").pipe(Effect.either))._tag).toBe("Left")
    expect((yield* owner.download.pipe(Effect.either))._tag).toBe("Left")
  })))

  it("reports source failures and permits a fresh explicit check", () => run(Effect.gen(function* () {
    const failed = yield* Ref.make(true)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Ref.get(failed).pipe(Effect.flatMap(value => value
        ? new ApplicationUpdateFailed({ message: "Registry unavailable" }) : Effect.succeed(Option.none()))),
      download: () => Effect.die("must not download"), stage: () => Effect.die("must not stage"),
    }))
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Failed"), Stream.take(1), Stream.runDrain)
    expect((yield* owner.state).check).toMatchObject({ _tag: "Failed", message: "Registry unavailable" })
    yield* Ref.set(failed, false)
    yield* owner.check("manual")
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
  })))

  it("never reports Ready when native signature verification fails", () => run(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: () => Effect.succeed(Option.some(candidate)), download: () => Effect.succeed("/verified/archive.zip"),
      stage: () => new ApplicationUpdateFailed({ message: "The native updater rejected this signature" }),
    }))
    yield* owner.check("manual")
    yield* waitFor(owner, "Available")
    yield* owner.download
    yield* waitFor(owner, "Failed")
    expect((yield* owner.requireReady.pipe(Effect.either))._tag).toBe("Left")
  })))

  describe("a newer release than the prepared update", () => {
    const newer = { ...candidate, version: "2.1.0" }
    const owner = (options: { readonly auto: boolean, readonly installation?: PreparedUpdate["installation"], readonly offers: Ref.Ref<Option.Option<typeof candidate>>,
      readonly download?: Effect.Effect<string, ApplicationUpdateFailed> }) => Effect.gen(function* () {
      const store = yield* PreparedUpdateStore
      const saved = yield* Ref.make(Option.some<PreparedUpdate>({ release: candidate, from: Option.some("1.0.0"), installation: options.installation ?? { _tag: "Unattempted" } }))
      const downloads = yield* Ref.make<string[]>([])
      const update = yield* makeApplicationUpdate(yield* Ref.get(saved)).pipe(
        Effect.provideService(PreparedUpdateStore, { ...store, read: Ref.get(saved) }),
        Effect.provideService(UpdatePreferences, { read: Effect.succeed(options.auto), write: () => Effect.void }),
        Effect.provideService(ApplicationUpdateSource, {
          check: () => Ref.get(options.offers),
          download: release => Ref.update(downloads, all => [...all, release.version]).pipe(Effect.zipRight(options.download ?? Effect.succeed("archive"))),
          stage: (_, release) => Ref.set(saved, Option.some({ release, from: Option.some("1.0.0"), installation: { _tag: "Unattempted" } })),
        }))
      return { update, saved, downloads }
    })
    const checked = (update: ApplicationUpdate) => update.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)

    it("downloads it automatically and replaces the prepared update", () => run(Effect.gen(function* () {
      const { update, saved, downloads } = yield* owner({ auto: true, offers: yield* Ref.make(Option.some(newer)) })
      expect((yield* update.state).transfer).toEqual({ _tag: "Ready", version: candidate.version })
      yield* update.check("scheduled")
      yield* update.changes.pipe(Stream.filter(state => state.transfer._tag === "Ready" && state.transfer.version === newer.version), Stream.take(1), Stream.runDrain)
      expect(yield* Ref.get(downloads)).toEqual([newer.version])
      expect(Option.getOrThrow(yield* Ref.get(saved)).release.version).toBe(newer.version)
      yield* update.requireReady
    })))
    it("offers it for download when automatic downloads are off", () => run(Effect.gen(function* () {
      const { update, downloads } = yield* owner({ auto: false, offers: yield* Ref.make(Option.some(newer)) })
      yield* update.check("scheduled")
      yield* checked(update)
      expect((yield* update.state).transfer).toEqual({ _tag: "Available", version: newer.version, bytes: newer.bytes, failed: Option.none() })
      expect(yield* Ref.get(downloads)).toEqual([])
      yield* update.download
      yield* update.changes.pipe(Stream.filter(state => state.transfer._tag === "Ready" && state.transfer.version === newer.version), Stream.take(1), Stream.runDrain)
      expect(yield* Ref.get(downloads)).toEqual([newer.version])
    })))
    it("supersedes a prepared update that failed to install", () => run(Effect.gen(function* () {
      const { update, downloads } = yield* owner({ auto: true, installation: { _tag: "Failed", kind: "install", reason: "The installer failed" }, offers: yield* Ref.make(Option.some(newer)) })
      expect((yield* update.state).transfer._tag).toBe("InstallationFailed")
      yield* update.check("scheduled")
      yield* update.changes.pipe(Stream.filter(state => state.transfer._tag === "Ready" && state.transfer.version === newer.version), Stream.take(1), Stream.runDrain)
      expect(yield* Ref.get(downloads)).toEqual([newer.version])
    })))
    it("keeps the prepared update installable when the newer download fails, and offers it again once the release is gone", () => run(Effect.gen(function* () {
      const offers = yield* Ref.make(Option.some(newer))
      const { update, saved } = yield* owner({ auto: true, offers, download: new ApplicationUpdateFailed({ message: "Network down" }) })
      yield* update.check("scheduled")
      yield* waitFor(update, "Failed")
      expect(Option.getOrThrow(yield* Ref.get(saved)).release.version).toBe(candidate.version)
      yield* Ref.set(offers, Option.none())
      yield* update.check("scheduled")
      yield* checked(update)
      expect((yield* update.state).transfer).toEqual({ _tag: "Ready", version: candidate.version })
      yield* update.requireReady
    })))
    it("does not download a newer version held after its own failure", () => run(Effect.gen(function* () {
      const store = yield* PreparedUpdateStore
      const { update, downloads } = yield* owner({ auto: true, offers: yield* Ref.make(Option.some(newer)) }).pipe(Effect.provideService(PreparedUpdateStore, {
        ...store, hold: Effect.succeed(Option.some({ version: Option.some(newer.version), reason: Option.some("install" as const), until: 0 })) }))
      yield* update.check("scheduled")
      yield* checked(update)
      expect((yield* update.state).transfer).toEqual({ _tag: "Available", version: newer.version, bytes: newer.bytes, failed: Option.some("install") })
      expect(yield* Ref.get(downloads)).toEqual([])
    })))
  })
  describe("after an unattended installation failed", () => {
    const owner = (hold: Ref.Ref<Option.Option<UnattendedHold>>, offer: typeof candidate, downloads: Ref.Ref<string[]>) => Effect.gen(function* () {
      const store = yield* PreparedUpdateStore
      return yield* makeApplicationUpdate().pipe(
        Effect.provideService(PreparedUpdateStore, { ...store, hold: Ref.get(hold), recordHold: value => Ref.set(hold, Option.some(value)), releaseHold: Ref.set(hold, Option.none()) }),
        Effect.provideService(UpdatePreferences, { read: Effect.succeed(true), write: () => Effect.void }),
        Effect.provideService(ApplicationUpdateSource, { check: () => Effect.succeed(Option.some(offer)),
          download: release => Ref.update(downloads, all => [...all, release.version]).pipe(Effect.as("archive")), stage: () => Effect.void }))
    })
    const checked = (update: ApplicationUpdate) => update.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)

    it.each(["launch", "scheduled"] as const)("does not download the failed version again on a %s check, in any later owner", reason => run(Effect.gen(function* () {
      const hold = yield* Ref.make(Option.some<UnattendedHold>({ version: Option.some(candidate.version), reason: Option.some("install"), until: 0 }))
      const downloads = yield* Ref.make<string[]>([])
      for (let restart = 0; restart < 3; restart++) {
        const update = yield* owner(hold, candidate, downloads)
        yield* update.check(reason)
        yield* checked(update)
        expect((yield* update.state).transfer).toEqual({ _tag: "Available", version: candidate.version, bytes: candidate.bytes, failed: Option.some("install") })
      }
      expect(yield* Ref.get(downloads)).toEqual([])
      expect(Option.isSome(yield* Ref.get(hold))).toBe(true)
    })))
    it("still downloads a newer version automatically", () => run(Effect.gen(function* () {
      const hold = yield* Ref.make(Option.some<UnattendedHold>({ version: Option.some(candidate.version), reason: Option.some("install"), until: 0 }))
      const downloads = yield* Ref.make<string[]>([])
      const update = yield* owner(hold, { ...candidate, version: "2.1.0" }, downloads)
      yield* update.check("scheduled")
      yield* waitFor(update, "Ready")
      expect(yield* Ref.get(downloads)).toEqual(["2.1.0"])
    })))
    it("lifts only the wait on a manual check: the failed version stays held and is not downloaded", () => run(Effect.gen(function* () {
      const hold = yield* Ref.make(Option.some<UnattendedHold>({ version: Option.some(candidate.version), reason: Option.some("startup"), until: Number.MAX_SAFE_INTEGER }))
      const downloads = yield* Ref.make<string[]>([])
      const update = yield* owner(hold, candidate, downloads)
      yield* update.check("manual")
      yield* checked(update)
      expect(yield* Ref.get(downloads)).toEqual([])
      expect(yield* Ref.get(hold)).toEqual(Option.some({ version: Option.some(candidate.version), reason: Option.some("startup"), until: 0 }))
    })))
    it("lets a person download it explicitly, lifting the hold", () => run(Effect.gen(function* () {
      const hold = yield* Ref.make(Option.some<UnattendedHold>({ version: Option.some(candidate.version), reason: Option.some("install"), until: Number.MAX_SAFE_INTEGER }))
      const downloads = yield* Ref.make<string[]>([])
      const update = yield* owner(hold, candidate, downloads)
      yield* update.check("launch")
      yield* waitFor(update, "Available")
      yield* update.download
      yield* waitFor(update, "Ready")
      expect(yield* Ref.get(downloads)).toEqual([candidate.version])
      expect(Option.isNone(yield* Ref.get(hold))).toBe(true)
    })))
  })
})
