import { randomUUID } from "node:crypto"
import { lstat, open, readFile, rename, unlink } from "node:fs/promises"
import { join } from "node:path"
import { Effect, Layer, Option, Schema } from "effect"
import { PrivateFilePermissions } from "./private-files"
import { PreviousInstallationFailed, PreviousInstallationJournal, PreviousInstallationPlan } from "./previous-installation"

const missing = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"
const failed = (error: unknown) => new PreviousInstallationFailed({ message: `Cannot access previous-installation recovery record: ${String(error)}` })

/** The caller owns the private application state directory and its native application lock. */
export const previousInstallationJournal = (directory: string) => Layer.effect(PreviousInstallationJournal, Effect.gen(function* () {
  const permissions = yield* PrivateFilePermissions
  const path = join(directory, "previous-installation.json")
  return PreviousInstallationJournal.of({
    read: Effect.tryPromise({ try: async () => {
      const info = await lstat(path).catch(error => { if (missing(error)) return null; throw error })
      if (info === null) return Option.none<string>()
      if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1 || info.size > 1024 * 1024 || info.uid !== process.getuid!()) throw new Error("Unsafe recovery record")
      return Option.some(await readFile(path, "utf8"))
    }, catch: failed }).pipe(Effect.flatMap(Option.match({
      onNone: () => Effect.succeed(Option.none<PreviousInstallationPlan>()),
      onSome: text => Schema.decodeUnknown(Schema.parseJson(PreviousInstallationPlan))(text).pipe(Effect.map(Option.some), Effect.mapError(failed)),
    }))),
    write: plan => Effect.gen(function* () {
      const text = yield* Schema.encode(Schema.parseJson(PreviousInstallationPlan))(plan).pipe(Effect.mapError(failed))
      const temporary = join(directory, `.previous-installation-${randomUUID()}.tmp`)
      yield* Effect.acquireUseRelease(permissions.createFile(temporary).pipe(Effect.mapError(failed)),
        () => Effect.tryPromise({ try: async () => {
          const file = await open(temporary, "r+")
          try { await file.writeFile(text); await file.sync() } finally { await file.close() }
          await rename(temporary, path)
          const parent = await open(directory, "r")
          try { await parent.sync() } finally { await parent.close() }
        }, catch: failed }),
        () => Effect.tryPromise({ try: () => unlink(temporary).catch(error => { if (!missing(error)) throw error }), catch: failed }).pipe(Effect.orDie))
    }).pipe(Effect.uninterruptible),
    clear: Effect.tryPromise({ try: () => unlink(path).catch(error => { if (!missing(error)) throw error }), catch: failed }),
  })
}))
