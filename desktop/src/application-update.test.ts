import { Deferred, Effect, Option, Ref, Schema, Scope, Stream } from "effect"
import { DesktopUpdateCandidate } from "@magnitudedev/release"
import { describe, expect, it } from "vitest"
import { ApplicationUpdateFailed, ApplicationUpdateSource, makeApplicationUpdate, type ApplicationUpdate } from "./application-update"

const candidate = Schema.decodeUnknownSync(DesktopUpdateCandidate)({ version: "2.0.0", artifact: {
  id: "desktop-update-darwin-arm64", kind: "desktop", host: "darwin-arm64",
  filename: "magnitude-desktop-darwin-arm64.zip", bytes: 100, sha256: "a".repeat(64),
} })
const waitFor = (owner: ApplicationUpdate, tag: string) => owner.changes.pipe(Stream.filter(state => state._tag === tag), Stream.take(1), Stream.runDrain)
const run = <A, E>(effect: Effect.Effect<A, E, Scope.Scope>) => Effect.runPromise(effect.pipe(Effect.scoped, Effect.timeout("3 seconds")))

describe("application-owned updates", () => {
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
    expect((yield* owner.check.pipe(Effect.either))._tag).toBe("Left")
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
    expect((yield* owner.state)._tag).toBe("Closed")
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
    yield* waitFor(owner, "Failed")
    expect(yield* owner.state).toMatchObject({ _tag: "Failed", message: "Registry unavailable" })
    yield* Ref.set(failed, false)
    yield* owner.check
    yield* waitFor(owner, "Current")
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
