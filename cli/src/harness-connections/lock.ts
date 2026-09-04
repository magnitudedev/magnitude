import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { BunSqliteDriverLayer } from "@magnitudedev/sdk/bun"
import { SqliteDriver } from "@magnitudedev/sdk/sqlite-driver"
import { Effect, Schedule } from "effect"

/** SQLite holds an OS-released lock across processes; elapsed time never steals ownership. */
export const withConnectionLock = <A, E, R>(manifest: string, effect: Effect.Effect<A, E, R>) => Effect.scoped(
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    yield* fs.makeDirectory(path.dirname(manifest), { recursive: true, mode: 0o700 })
    const driver = yield* SqliteDriver
    const database = yield* driver.open(`${manifest}.lock.sqlite`, { create: true })
    yield* Effect.uninterruptibleMask((restore) => Effect.acquireRelease(
      restore(database.execute("BEGIN IMMEDIATE").pipe(Effect.retry({
        while: (error) => error._tag === "SqliteDriverBusy",
        schedule: Schedule.spaced("50 millis"),
      }))),
      () => database.execute("ROLLBACK").pipe(Effect.orDie),
    ))
    return yield* effect
  }).pipe(Effect.provide(BunSqliteDriverLayer)),
)
