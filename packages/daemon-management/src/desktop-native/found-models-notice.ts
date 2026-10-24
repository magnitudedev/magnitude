import { FileSystem } from "@effect/platform"
import { GlobalStorage, makeGlobalStorage, makeConfigStorage, MagnitudeConfigSchema, readStructuredFile } from "@magnitudedev/storage"
import { Context, Effect, Option, Schema } from "effect"

export class FoundModelsNoticeFailed extends Schema.TaggedError<FoundModelsNoticeFailed>()("FoundModelsNoticeFailed", {
  message: Schema.String,
}) {}

export interface FoundModelsNoticePreferences {
  readonly readDismissed: Effect.Effect<boolean, FoundModelsNoticeFailed>
  readonly dismiss: Effect.Effect<void, FoundModelsNoticeFailed>
}
export const FoundModelsNoticePreferences = Context.GenericTag<FoundModelsNoticePreferences>("desktop/FoundModelsNoticePreferences")

export const makeFoundModelsNoticePreferences = (clientDataDirectory: string) => Effect.gen(function* () {
  const fs = yield* FileSystem.FileSystem
  const storage = makeGlobalStorage({ root: clientDataDirectory })
  const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
  return FoundModelsNoticePreferences.of({
    readDismissed: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("foundModelsNoticeDismissed")).pipe(
      Effect.provideService(FileSystem.FileSystem, fs),
      Effect.flatMap(result => result._tag === "Invalid" ? Effect.fail(result.error)
        : Effect.succeed(result._tag === "Missing" ? false : Option.getOrElse(result.value.foundModelsNoticeDismissed, () => false))),
      Effect.mapError(() => new FoundModelsNoticeFailed({ message: "The saved model notice preference could not be read." })),
    ),
    dismiss: config.update(current => ({ ...current, foundModelsNoticeDismissed: Option.some(true) })).pipe(
      Effect.asVoid,
      Effect.mapError(() => new FoundModelsNoticeFailed({ message: "The model notice could not be dismissed. Check access to the Magnitude configuration and try again." })),
    ),
  })
})
