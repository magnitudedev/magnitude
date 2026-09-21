import { FileSystem } from "@effect/platform"
import { GlobalStorage, makeGlobalStorage, makeConfigStorage, MagnitudeConfigSchema, readStructuredFile, resolveModelStoreLocation, selectModelStorePath, type ModelStoreLocation } from "@magnitudedev/storage"
import { Context, Effect, Option, Schema } from "effect"
import { isAbsolute } from "node:path"

export class ModelStoragePreferencesFailed extends Schema.TaggedError<ModelStoragePreferencesFailed>()("ModelStoragePreferencesFailed", {
  message: Schema.String,
}) {}

export interface ModelStoragePreferences {
  readonly defaultPath: string
  /** What the next service start will use, resolved exactly as the service resolves it. */
  readonly read: Effect.Effect<ModelStoreLocation & { readonly defaultPath: string }, ModelStoragePreferencesFailed>
  /** `None` removes the setting so the default applies again. */
  readonly write: (path: Option.Option<string>) => Effect.Effect<void, ModelStoragePreferencesFailed>
}
export const ModelStoragePreferences = Context.GenericTag<ModelStoragePreferences>("desktop/ModelStoragePreferences")

export const makeModelStoragePreferences = (clientDataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const storage = makeGlobalStorage({ root: clientDataDirectory })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  const defaultPath = selectModelStorePath(clientDataDirectory, Option.none()).path
  return ModelStoragePreferences.of({
    defaultPath,
    read: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("modelsDirectory")).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error)
        : resolveModelStoreLocation(clientDataDirectory, result._tag === "Missing" ? Option.none() : result.value.modelsDirectory).pipe(
          Effect.provideService(FileSystem.FileSystem, fs))),
      Effect.map(location => ({ ...location, defaultPath })),
      Effect.mapError(() => new ModelStoragePreferencesFailed({ message: "The saved model storage location could not be read. Using the default folder." })),
    ),
    write: path => Effect.gen(function* () {
      const trimmed = Option.map(path, value => value.trim())
      if (Option.isSome(trimmed) && (trimmed.value.length === 0 || !isAbsolute(trimmed.value))) {
        return yield* new ModelStoragePreferencesFailed({ message: "Choose a full folder path for model storage." })
      }
      yield* config.update(current => ({ ...current, modelsDirectory: trimmed })).pipe(Effect.mapError(() => new ModelStoragePreferencesFailed({ message: "The model storage location could not be saved. Check access to the Magnitude configuration and try again." })))
    }),
  })
})
