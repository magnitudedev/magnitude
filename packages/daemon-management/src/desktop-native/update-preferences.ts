import { FileSystem } from "@effect/platform"
import { GlobalStorage, makeGlobalStorage, makeConfigStorage, MagnitudeConfigSchema, readStructuredFile } from "@magnitudedev/storage"
import { Context, Effect, Option, Schema } from "effect"

export class UpdatePreferencesFailed extends Schema.TaggedError<UpdatePreferencesFailed>()("UpdatePreferencesFailed", {
  message: Schema.String,
}) {}

export interface UpdatePreferences {
  readonly read: Effect.Effect<boolean, UpdatePreferencesFailed>
  readonly write: (autoDownload: boolean) => Effect.Effect<void, UpdatePreferencesFailed>
}
export const UpdatePreferences = Context.GenericTag<UpdatePreferences>("@magnitudedev/daemon-management/UpdatePreferences")

/** The application owns the preference, using the same canonical config schema and atomic writer as other settings. */
export const makeUpdatePreferences = (clientDataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const storage = makeGlobalStorage({ root: clientDataDirectory })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  return UpdatePreferences.of({
    read: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("autoDownloadUpdates")).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error)
        : Effect.succeed(result._tag === "Missing" ? true : Option.getOrElse(result.value.autoDownloadUpdates, () => true))),
      Effect.mapError(() => new UpdatePreferencesFailed({ message: "Update preferences could not be read. Automatic downloads are paused." })),
    ),
    write: enabled => config.update(current => ({ ...current, autoDownloadUpdates: Option.some(enabled) })).pipe(
      Effect.asVoid,
      Effect.mapError(() => new UpdatePreferencesFailed({ message: "Update preferences could not be saved." })),
    ),
  })
})
