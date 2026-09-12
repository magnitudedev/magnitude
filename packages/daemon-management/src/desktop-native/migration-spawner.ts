import { Effect } from "effect"
import { type LegacyMigrationFailed } from "./legacy-migration"
import { OwnedChildSpawner, OwnedChildSpawnFailed } from "./owned-child"

/** Admission belongs to the supervisor fiber: the tray and control server remain responsive. */
export const migrateBeforeSpawn = (migration: Effect.Effect<void, LegacyMigrationFailed>) => Effect.gen(function* () {
  const spawner = yield* OwnedChildSpawner
  return OwnedChildSpawner.of({
    spawn: command => migration.pipe(
      Effect.mapError(error => new OwnedChildSpawnFailed({ executable: command.executable, message: error.message })),
      Effect.zipRight(spawner.spawn(command)),
    ),
  })
})
