import { Deferred, Effect, Option, Ref, Schema, Scope, Stream } from "effect"
import { UpdateManifest } from "@magnitudedev/release/hosted-update"
import { UpdatePreferences, UpdatePreferencesFailed } from "./update-preferences"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate, type ApplicationUpdate } from "./application-update"

const candidate = Schema.decodeUnknownSync(UpdateManifest)({ protocol: 1, version: "2.0.0", commit: "a".repeat(40), artifact: {
  id: "desktop-update-darwin-arm64", target: { os: "darwin", arch: "arm64", package: "mac-zip" },
  path: "releases/2.0.0/magnitude-desktop-darwin-arm64.zip", bytes: 100, sha256: "a".repeat(64),
} })
const waitFor = (owner: ApplicationUpdate, tag: string) => owner.changes.pipe(Stream.filter(state => state.transfer._tag === tag), Stream.take(1), Stream.runDrain)
const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope | UpdatePreferences>) => Effect.runPromise(effect.pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(false), write: () => Effect.void }), Effect.scoped, Effect.timeout("3 seconds")))

describe("application-owned updates", () => {
  it("continues checking after a preference read failure and never claims an unsaved choice", () => run(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(UpdatePreferences, {
      read: new UpdatePreferencesFailed({ message: "Unreadable preferences" }),
      write: () => new UpdatePreferencesFailed({ message: "Read-only profile" }),
    }), Effect.provideService(ApplicationUpdateSource, {
      check: Effect.succeed(Option.some(candidate)), download: () => Effect.die("Automatic download must remain paused"), stage: () => Effect.void,
    }))
    yield* owner.check
    yield* waitFor(owner, "Available")
    expect((yield* owner.setAutoDownload(true).pipe(Effect.either))._tag).toBe("Left")
    expect((yield* owner.state).preference._tag).toBe("Unavailable")
    expect((yield* owner.state).transfer._tag).toBe("Available")
  })))
  it("keeps checking while an automatic download is active and after staging without replacing it", () => run(Effect.gen(function* () {
    const checks = yield* Ref.make(0)
    const finish = yield* Deferred.make<void>()
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(UpdatePreferences, { read: Effect.succeed(true), write: () => Effect.void }), Effect.provideService(ApplicationUpdateSource, {
      check: Ref.updateAndGet(checks, n => n + 1).pipe(Effect.map(n => n === 1 ? Option.some(candidate) : Option.none())),
      download: () => Deferred.await(finish).pipe(Effect.as("archive")), stage: () => Effect.void,
    }))
    yield* owner.check
    yield* waitFor(owner, "Downloading")
    yield* owner.check
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
    expect(yield* Ref.get(checks)).toBe(2)
    expect((yield* owner.state).transfer._tag).toBe("Downloading")
    yield* Deferred.succeed(finish, undefined)
    yield* waitFor(owner, "Ready")
    yield* owner.check
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
      check: Effect.succeed(Option.some(candidate)),
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
    yield* owner.check
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
      check: Effect.succeed(Option.some(candidate)), download: () => Deferred.await(finish).pipe(Effect.as("archive")), stage: () => Effect.void,
    }))
    yield* owner.check
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
      check: Ref.update(checks, n => n + 1).pipe(Effect.zipRight(Deferred.await(finishCheck)), Effect.as(Option.some(candidate))),
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
    yield* owner.check
    expect((yield* owner.check.pipe(Effect.either))._tag).toBe("Right")
    yield* Deferred.succeed(finishCheck, undefined)
    yield* waitFor(owner, "Available")
    expect(yield* Ref.get(checks)).toBe(1)
    yield* owner.download
    yield* waitFor(owner, "Staging")
    expect((yield* owner.requireReady.pipe(Effect.either))._tag).toBe("Left")
    expect((yield* owner.download.pipe(Effect.either))._tag).toBe("Left")
    // All observers above have ended. Their lifetime cannot end staging.
    yield* Deferred.succeed(finishStage, undefined)
    yield* waitFor(owner, "Ready")
    yield* owner.requireReady
    yield* owner.close
    expect(yield* Ref.get(released)).toBe(true)
    expect((yield* owner.state).transfer._tag).toBe("Closed")
  })))

  it("Quit cancels an unfinished transfer, closes admission, and never stages it", () => run(Effect.gen(function* () {
    const started = yield* Deferred.make<void>()
    const cancelled = yield* Deferred.make<void>()
    const staged = yield* Ref.make(false)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: Effect.succeed(Option.some(candidate)),
      download: () => Deferred.succeed(started, undefined).pipe(Effect.zipRight(Effect.never), Effect.ensuring(Deferred.succeed(cancelled, undefined))),
      stage: () => Ref.set(staged, true),
    }))
    yield* owner.check
    yield* waitFor(owner, "Available")
    yield* owner.download
    yield* Deferred.await(started)
    yield* owner.close
    yield* Deferred.await(cancelled)
    expect(yield* Ref.get(staged)).toBe(false)
    expect((yield* owner.check.pipe(Effect.either))._tag).toBe("Left")
    expect((yield* owner.download.pipe(Effect.either))._tag).toBe("Left")
  })))

  it("reports source failures and permits a fresh explicit check", () => run(Effect.gen(function* () {
    const failed = yield* Ref.make(true)
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: Ref.get(failed).pipe(Effect.flatMap(value => value
        ? new ApplicationUpdateFailed({ message: "Registry unavailable" }) : Effect.succeed(Option.none()))),
      download: () => Effect.die("must not download"), stage: () => Effect.die("must not stage"),
    }))
    yield* owner.check
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Failed"), Stream.take(1), Stream.runDrain)
    expect((yield* owner.state).check).toMatchObject({ _tag: "Failed", message: "Registry unavailable" })
    yield* Ref.set(failed, false)
    yield* owner.check
    yield* owner.changes.pipe(Stream.filter(state => state.check._tag === "Succeeded"), Stream.take(1), Stream.runDrain)
  })))

  it("never reports Ready when native signature verification fails", () => run(Effect.gen(function* () {
    const owner = yield* makeApplicationUpdate().pipe(Effect.provideService(ApplicationUpdateSource, {
      check: Effect.succeed(Option.some(candidate)), download: () => Effect.succeed("/verified/archive.zip"),
      stage: () => new ApplicationUpdateFailed({ message: "The native updater rejected this signature" }),
    }))
    yield* owner.check
    yield* waitFor(owner, "Available")
    yield* owner.download
    yield* waitFor(owner, "Failed")
    expect((yield* owner.requireReady.pipe(Effect.either))._tag).toBe("Left")
  })))
})
