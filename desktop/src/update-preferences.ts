import { FileSystem } from "@effect/platform"
import { Context, Effect, Schema } from "effect"
import { join } from "node:path"

const Preferences = Schema.Struct({ autoDownload: Schema.Boolean })
const PreferencesJson = Schema.parseJson(Preferences)

export class UpdatePreferencesFailed extends Schema.TaggedError<UpdatePreferencesFailed>()("UpdatePreferencesFailed", {
  message: Schema.String,
}) {}

export interface UpdatePreferences {
  readonly read: Effect.Effect<boolean, UpdatePreferencesFailed>
  readonly write: (autoDownload: boolean) => Effect.Effect<void, UpdatePreferencesFailed>
}
export const UpdatePreferences = Context.GenericTag<UpdatePreferences>("desktop/UpdatePreferences")

/** The desktop owner persists this independently of service availability. */
export const makeUpdatePreferences = (clientStateDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const directory = join(clientStateDirectory, "updates")
  const path = join(directory, "preferences.json")
  const gate = yield* Effect.makeSemaphore(1)
  return UpdatePreferences.of({
    read: Effect.gen(function* () {
      if (!(yield* fs.exists(path))) return true
      const stat = yield* fs.stat(path)
      if (stat.type !== "File" || stat.size > 4096) return yield* new UpdatePreferencesFailed({ message: "Invalid update preferences" })
      return (yield* Schema.decodeUnknown(PreferencesJson)(yield* fs.readFileString(path))).autoDownload
    }).pipe(Effect.mapError(() => new UpdatePreferencesFailed({ message: "Update preferences could not be read. Automatic downloads are paused." }))),
    write: autoDownload => gate.withPermits(1)(Effect.scoped(Effect.gen(function* () {
      const contents = yield* Schema.encode(PreferencesJson)({ autoDownload })
      yield* fs.makeDirectory(directory, { recursive: true, mode: 0o700 })
      const temporary = yield* fs.makeTempDirectoryScoped({ directory, prefix: "preferences-" })
      const pending = join(temporary, "preferences.json")
      yield* fs.writeFileString(pending, contents, { mode: 0o600, flag: "wx" })
      yield* fs.rename(pending, path)
    })).pipe(Effect.uninterruptible, Effect.mapError(() => new UpdatePreferencesFailed({ message: "Update preferences could not be saved." })))),
  })
})
