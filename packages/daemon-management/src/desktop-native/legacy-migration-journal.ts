import { randomUUID } from "node:crypto"
import { lstat, mkdir, open, readFile, rename, unlink } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Layer, Option, Schema } from "effect"
import { LegacyMigrationFailed, LegacyMigrationJournal, LegacyMigrationState } from "./legacy-migration"

const absent = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"
const failed = (error: unknown) => new LegacyMigrationFailed({ message: `Could not access legacy migration checkpoint: ${String(error)}` })

/** The private application directory and passive application lock are owned by the caller. */
export const fileLegacyMigrationJournal = (directory: string) => Layer.succeed(LegacyMigrationJournal, {
  read: Effect.tryPromise({ try: async () => {
    const path = join(directory, "legacy-migration.json")
    const info = await lstat(path).catch(error => { if (absent(error)) return null; throw error })
    if (info === null) return Option.none<string>()
    if (!info.isFile() || info.isSymbolicLink() || info.size > 1024 * 1024 || (process.platform !== "win32" && info.uid !== process.getuid!())) throw new Error("Migration checkpoint is not a safe user-owned file")
    return Option.some(await readFile(path, "utf8"))
  }, catch: failed }).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.succeed(Option.none<LegacyMigrationState>()),
    onSome: text => Schema.decodeUnknown(Schema.parseJson(LegacyMigrationState))(text).pipe(Effect.map(Option.some),
      Effect.mapError(() => new LegacyMigrationFailed({ message: "Legacy migration checkpoint is malformed; it has not been reset" }))),
  }))),
  write: state => Schema.encode(Schema.parseJson(LegacyMigrationState))(state).pipe(
    Effect.mapError(failed),
    Effect.flatMap(text => Effect.tryPromise({ try: async () => {
      await mkdir(directory, { recursive: true, mode: 0o700 })
      const temporary = join(directory, `.legacy-migration-${randomUUID()}.tmp`)
      try {
        const file = await open(temporary, "wx", 0o600)
        try { await file.writeFile(text); await file.sync() } finally { await file.close() }
        await rename(temporary, join(directory, "legacy-migration.json"))
        if (process.platform !== "win32") {
          const parent = await open(directory, "r")
          try { await parent.sync() } finally { await parent.close() }
        }
      } finally { await unlink(temporary).catch(error => { if (!absent(error)) throw error }) }
    }, catch: failed })),
  ).pipe(Effect.uninterruptible),
})
