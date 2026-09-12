import { Deferred, Effect, Ref, TestClock, TestContext } from "effect"
import { describe, expect, it } from "vitest"
import { LegacyMigrationFailed } from "./legacy-migration"
import { migrateBeforeSpawn } from "./migration-spawner"
import { OwnedChildSpawner } from "./owned-child"
import { makeOwnedService } from "./owned-service"

const setup = (migration: Effect.Effect<void, LegacyMigrationFailed>) => Effect.gen(function* () {
  const launches = yield* Ref.make(0)
  const spawner = yield* migrateBeforeSpawn(migration).pipe(Effect.provideService(OwnedChildSpawner, {
    spawn: () => Ref.update(launches, count => count + 1).pipe(Effect.zipRight(Effect.never)),
  }))
  const service = yield* makeOwnedService({ executable: "fixture", arguments: [], environment: {} }, 1)
    .pipe(Effect.provideService(OwnedChildSpawner, spawner))
  return { service, launches }
})
const run = <A, E>(effect: Effect.Effect<A, E, import("effect").Scope.Scope>) =>
  Effect.runPromise(Effect.scoped(effect).pipe(Effect.provide(TestContext.TestContext)))

describe("migration admission in the service supervisor", () => {
  it("keeps observation available and cannot spawn after Quit during migration", () => run(Effect.gen(function* () {
    const gate = yield* Deferred.make<void>()
    const f = yield* setup(Deferred.await(gate))
    expect((yield* f.service.state)._tag).toBe("Starting")
    expect(yield* Ref.get(f.launches)).toBe(0)
    yield* f.service.shutdown
    yield* Deferred.succeed(gate, undefined)
    yield* TestClock.adjust("1 minute")
    expect((yield* f.service.state)._tag).toBe("Stopped")
    expect(yield* Ref.get(f.launches)).toBe(0)
  })))
  it("presents migration failure, bounds retries, and admits only after explicit successful retry", () => run(Effect.gen(function* () {
    const repaired = yield* Ref.make(false)
    const attempts = yield* Ref.make(0)
    const migration = Ref.update(attempts, count => count + 1).pipe(Effect.zipRight(
      Ref.get(repaired).pipe(Effect.flatMap(value => value ? Effect.void : Effect.fail(new LegacyMigrationFailed({ message: "Legacy ownership changed" }))))))
    const f = yield* setup(migration)
    yield* TestClock.adjust("8 seconds")
    expect(yield* f.service.state).toMatchObject({ _tag: "Failed", message: "Legacy ownership changed" })
    expect(yield* Ref.get(attempts)).toBe(4)
    expect(yield* Ref.get(f.launches)).toBe(0)
    yield* Ref.set(repaired, true)
    yield* TestClock.adjust("1 minute")
    expect(yield* Ref.get(f.launches)).toBe(0)
    yield* f.service.retry
    yield* TestClock.adjust("1 millis")
    expect(yield* Ref.get(f.launches)).toBe(1)
    yield* f.service.shutdown
  })))
})
