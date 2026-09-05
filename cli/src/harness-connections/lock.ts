import * as FileSystem from "@effect/platform/FileSystem"
import * as Path from "@effect/platform/Path"
import { BunSqliteDriverLayer } from "@magnitudedev/daemon-management/bun"
import { SqliteDriver } from "@magnitudedev/daemon-management/sqlite-driver"
import { Duration, Effect, Schedule, Schema } from "effect"

export class HarnessConnectionLockTimedOut extends Schema.TaggedError<HarnessConnectionLockTimedOut>()(
  "HarnessConnectionLockTimedOut",
  { path: Schema.String, waited: Schema.String },
) {
  override get message(): string {
    return `Another Magnitude process held the harness connection lock ${this.path} for ${this.waited}`
  }
}

export const HARNESS_CONNECTION_LOCK_BOUND = Duration.seconds(30)

/**
 * SQLite holds an OS-released lock across processes; elapsed time never steals
 * ownership. Waiting for it is bounded so a holder that never finishes surfaces
 * as an error naming the lock instead of an indefinite wait.
 */
export const withConnectionLock = <A, E, R>(
  manifest: string,
  effect: Effect.Effect<A, E, R>,
  bound: Duration.DurationInput = HARNESS_CONNECTION_LOCK_BOUND,
) => Effect.scoped(
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const path = yield* Path.Path
    yield* fs.makeDirectory(path.dirname(manifest), { recursive: true, mode: 0o700 })
    const driver = yield* SqliteDriver
    const lock = `${manifest}.lock.sqlite`
    const database = yield* driver.open(lock, { create: true })
    yield* Effect.uninterruptibleMask((restore) => Effect.acquireRelease(
      restore(database.execute("BEGIN IMMEDIATE").pipe(
        Effect.retry({
          while: (error) => error._tag === "SqliteDriverBusy",
          schedule: Schedule.spaced("50 millis"),
        }),
        Effect.timeoutFail({
          duration: bound,
          onTimeout: () => new HarnessConnectionLockTimedOut({
            path: lock,
            waited: Duration.format(Duration.decode(bound)),
          }),
        }),
      )),
      () => database.execute("ROLLBACK").pipe(Effect.orDie),
    ))
    return yield* effect
  }).pipe(Effect.provide(BunSqliteDriverLayer)),
)
