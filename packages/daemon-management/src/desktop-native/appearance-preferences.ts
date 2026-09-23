import { FileSystem } from "@effect/platform"
import { GlobalStorage, makeGlobalStorage, makeConfigStorage, MagnitudeConfigSchema, readStructuredFile } from "@magnitudedev/storage"
import type { AppearancePreference } from "@magnitudedev/sdk/desktop-host"
import { Context, Effect, Option, Schema } from "effect"

export class AppearancePreferencesFailed extends Schema.TaggedError<AppearancePreferencesFailed>()("AppearancePreferencesFailed", {
  message: Schema.String,
}) {}

export interface AppearancePreferences {
  readonly read: Effect.Effect<AppearancePreference, AppearancePreferencesFailed>
  readonly write: (preference: AppearancePreference) => Effect.Effect<void, AppearancePreferencesFailed>
}
export const AppearancePreferences = Context.GenericTag<AppearancePreferences>("desktop/AppearancePreferences")

export const makeAppearancePreferences = (clientDataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const storage = makeGlobalStorage({ root: clientDataDirectory })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  return AppearancePreferences.of({
    read: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("appearance")).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error)
        : Effect.succeed(result._tag === "Missing" ? "system" as const : Option.getOrElse(result.value.appearance, () => "system" as const))),
      Effect.mapError(() => new AppearancePreferencesFailed({ message: "The saved appearance could not be read. Using System appearance." })),
    ),
    write: preference => config.update(current => ({ ...current, appearance: Option.some(preference) })).pipe(
      Effect.asVoid,
      Effect.mapError(() => new AppearancePreferencesFailed({ message: "Appearance could not be saved. Check access to the Magnitude configuration and try again." })),
    ),
  })
})
