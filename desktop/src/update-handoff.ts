import { randomUUID } from "node:crypto"
import { lstat, open, readFile, rename, unlink } from "node:fs/promises"
import { join } from "node:path"
import { Context, Effect, Option, Schema } from "effect"
import { isNewerVersion, isValidVersion } from "@magnitudedev/release"
import { MacApplicationInstallation } from "@magnitudedev/daemon-management/desktop-native"

export class UpdateHandoffFailed extends Schema.TaggedError<UpdateHandoffFailed>()("UpdateHandoffFailed", { message: Schema.String }) {}
const Version = Schema.NonEmptyString.pipe(Schema.filter(isValidVersion))
const Receipt = Schema.Struct({ version: Version, bundle: Schema.NonEmptyString })
const Admission = Schema.Union(Schema.TaggedStruct("Continue", {}), Schema.TaggedStruct("Defer", {}), Schema.TaggedStruct("Failed", { message: Schema.String }))
export interface ApplicationUpdateHandoff {
  readonly record: (version: string) => Effect.Effect<void, UpdateHandoffFailed>
  readonly inspect: (version: string) => Effect.Effect<typeof Admission.Type, UpdateHandoffFailed>
}
export const ApplicationUpdateHandoff = Context.GenericTag<ApplicationUpdateHandoff>("desktop/ApplicationUpdateHandoff")
const absent = (error: unknown) => error instanceof Error && "code" in error && error.code === "ENOENT"
const failure = () => new UpdateHandoffFailed({ message: "Could not read or save the application update handoff. Check access to your Magnitude data directory." })

/** Called under the application lock. This receipt records update intent, never process liveness. */
export const makeUpdateHandoff = (directory: string, bundle: string) => Effect.gen(function* () {
  const native = yield* MacApplicationInstallation
  const path = join(directory, "update-handoff.json")
  const read = Effect.tryPromise({ try: async () => {
    const info = await lstat(path).catch(error => { if (absent(error)) return null; throw error })
    if (info === null) return Option.none<string>()
    if (!info.isFile() || info.isSymbolicLink() || info.size > 16 * 1024 || info.uid !== process.getuid!()) throw new Error("Unsafe update receipt")
    return Option.some(await readFile(path, "utf8"))
  }, catch: failure }).pipe(Effect.flatMap(Option.match({
    onNone: () => Effect.succeed(Option.none<typeof Receipt.Type>()),
    onSome: text => Schema.decodeUnknown(Schema.parseJson(Receipt))(text).pipe(Effect.map(Option.some), Effect.mapError(failure)),
  })))
  return ApplicationUpdateHandoff.of({
    record: version => Schema.encode(Schema.parseJson(Receipt))({ version, bundle }).pipe(Effect.mapError(failure), Effect.flatMap(text => Effect.tryPromise({ try: async () => {
      const temporary = join(directory, `.update-handoff-${randomUUID()}.tmp`)
      try {
        const file = await open(temporary, "wx", 0o600)
        try { await file.writeFile(text); await file.sync() } finally { await file.close() }
        await rename(temporary, path)
        const parent = await open(directory, "r")
        try { await parent.sync() } finally { await parent.close() }
      } finally { await unlink(temporary).catch(error => { if (!absent(error)) throw error }) }
    }, catch: failure })), Effect.uninterruptible),
    inspect: version => Effect.gen(function* () {
      const receipt = yield* read
      if (Option.isNone(receipt)) return { _tag: "Continue" as const }
      if (!isValidVersion(version)) return yield* new UpdateHandoffFailed({ message: "Could not identify the running application version during update recovery." })
      if (!isNewerVersion(receipt.value.version, version)) {
        yield* Effect.tryPromise({ try: () => unlink(path), catch: failure })
        return { _tag: "Continue" as const }
      }
      if (yield* native.isInstalling(receipt.value.bundle).pipe(Effect.mapError(error => new UpdateHandoffFailed({ message: error.message })))) return { _tag: "Defer" as const }
      return { _tag: "Failed" as const, message: `The previous update to ${receipt.value.version} did not finish. Check for updates to try again.` }
    }),
  })
})
