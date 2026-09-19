import { Deferred, Effect, Fiber, Layer, Option, Schema } from "effect"
import { expect, test } from "vitest"
import { installationSession } from "../src/installation-session"
import { Candidate } from "../src/candidate"
import { targets } from "../src/catalog"
import { Installer, InstalledApplication } from "../src/installer"
import { AssertionFailure } from "../src/domain"

const target = targets.find(target => target.id === "macos-15-arm64-metal-apple-silicon")!
const candidate = Schema.decodeUnknownSync(Candidate)({ version: "0.1.3", target, path: "/candidate.dmg",
  artifact: { id: "desktop-darwin-arm64", kind: "desktop", host: "darwin-arm64", filename: "Magnitude.dmg", bytes: 1, sha256: "a".repeat(64) } })
for (const scenario of ["reinstall", "remove", "remove-fails"] as const) test(`tracks installation ownership for ${scenario}`, async () => {
  let installs = 0, removals = 0
  const cleanupErrors: string[] = []
  const installer = Layer.succeed(Installer, {
    install: () => Effect.sync(() => { installs++; return InstalledApplication.make({ candidate, root: "/app", executable: "/app/exe", cli: "/app/cli", packageVersion: "0.1.3" }) }),
    uninstall: () => Effect.suspend(() => { removals++; return scenario === "remove-fails" ? Effect.fail(new AssertionFailure({ message: "Removal failed" })) : Effect.void }),
  })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const session = yield* installationSession(candidate, detail => { cleanupErrors.push(detail) })
    yield* Effect.all([session.get, session.get], { concurrency: "unbounded" })
    expect(installs).toBe(1)
    const removed = yield* session.remove.pipe(Effect.either)
    expect(removed._tag).toBe(scenario === "remove-fails" ? "Left" : "Right")
    if (scenario === "reinstall") yield* session.get
  })).pipe(Effect.provide(installer)))
  expect(installs).toBe(scenario === "reinstall" ? 2 : 1)
  expect(removals).toBe(scenario === "remove" ? 1 : 2)
  expect(cleanupErrors).toEqual(scenario === "remove-fails" ? ["Removal failed"] : [])
})

for (const scenario of ["replace", "remove-fails-once", "install-fails-once", "foreign-target"] as const) test(`replacement preserves exact ownership for ${scenario}`, async () => {
  const previous = Candidate.make({ ...candidate, version: "0.1.2", path: "/previous.dmg", artifact: { ...candidate.artifact, sha256: "b".repeat(64) } })
  const events: string[] = [], cleanupErrors: string[] = []
  let failed = false
  const installer = Layer.succeed(Installer, {
    install: value => Effect.suspend(() => {
      events.push(`install:${value.version}`)
      if (scenario === "install-fails-once" && value.version === previous.version && !failed) {
        failed = true
        return Effect.fail(new AssertionFailure({ message: "Install failed" }))
      }
      return Effect.succeed(InstalledApplication.make({ candidate: value, root: "/app", executable: "/app/exe", cli: "/app/cli", packageVersion: value.version }))
    }),
    uninstall: value => Effect.suspend(() => {
      events.push(`remove:${value.packageVersion}`)
      if (scenario === "remove-fails-once" && !failed) {
        failed = true
        return Effect.fail(new AssertionFailure({ message: "Removal failed" }))
      }
      return Effect.void
    }),
  })
  await Effect.runPromise(Effect.scoped(Effect.gen(function* () {
    const session = yield* installationSession(candidate, detail => { cleanupErrors.push(detail) })
    yield* session.get
    if (scenario === "foreign-target") {
      expect((yield* session.replace({ ...previous, target: { ...target, arch: "x64" } }).pipe(Effect.either))._tag).toBe("Left")
      expect(events).toEqual(["install:0.1.3"])
    } else if (scenario === "replace") {
      yield* Effect.all([session.replace(previous), session.replace(previous)], { concurrency: "unbounded" })
      expect(events).toEqual(["install:0.1.3", "remove:0.1.3", "install:0.1.2"])
      expect((yield* session.get).packageVersion).toBe("0.1.2")
      yield* session.remove
      expect((yield* session.get).packageVersion).toBe("0.1.2")
    } else {
      expect((yield* session.replace(previous).pipe(Effect.either))._tag).toBe("Left")
      if (scenario === "remove-fails-once") expect((yield* session.get).packageVersion).toBe("0.1.3")
      else {
        const before = events.length
        yield* session.remove
        expect(events.length).toBe(before)
        expect((yield* session.get).packageVersion).toBe("0.1.2")
      }
    }
  })).pipe(Effect.provide(installer)))
  expect(cleanupErrors).toEqual([])
  expect(events.at(-1)).toBe(scenario === "remove-fails-once" || scenario === "foreign-target" ? "remove:0.1.3" : "remove:0.1.2")
})


test("cancellation waits for an admitted replacement and preserves its cleanup ownership", () => Effect.runPromise(Effect.gen(function* () {
  const entered = yield* Deferred.make<void>(), release = yield* Deferred.make<void>()
  const previous = Candidate.make({ ...candidate, version: "0.1.2", path: "/previous.dmg" })
  const removed: string[] = []
  yield* Effect.scoped(Effect.gen(function* () {
    const session = yield* installationSession(candidate, () => {})
    yield* session.get
    const replacement = yield* Effect.forkScoped(session.replace(previous))
    yield* Deferred.await(entered)
    yield* Fiber.interruptFork(replacement)
    yield* Deferred.succeed(release, undefined)
    yield* Fiber.await(replacement)
    expect((yield* session.get).packageVersion).toBe("0.1.2")
  })).pipe(Effect.provideService(Installer, {
    install: value => Effect.gen(function* () {
      if (value.version === previous.version) {
        yield* Deferred.succeed(entered, undefined)
        yield* Deferred.await(release)
      }
      return InstalledApplication.make({ candidate: value, root: "/app", executable: "/app/exe", cli: "/app/cli", packageVersion: value.version })
    }),
    uninstall: value => Effect.sync(() => { removed.push(value.packageVersion) }),
  }))
  expect(removed).toEqual(["0.1.3", "0.1.2"])
})))
