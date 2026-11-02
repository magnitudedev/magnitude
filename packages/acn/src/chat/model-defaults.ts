import { FileSystem } from "@effect/platform"
import { Context, Effect, Layer, Option, Schema } from "effect"
import {
  Chat,
  ChatModelDefaults,
  ChatModelDefaultsFailed,
  type SavedChatModelDefaults,
  type ServingModelId,
} from "@magnitudedev/acn-protocol"
import {
  GlobalStorage,
  MagnitudeConfigSchema,
  makeConfigStorage,
  makeGlobalStorage,
  readStructuredFile,
} from "@magnitudedev/storage"
import { AcnChanges } from "../changes"
import { AcnHost } from "../server-settings"

export interface ChatModelDefaultsStoreApi {
  readonly list: Effect.Effect<SavedChatModelDefaults, ChatModelDefaultsFailed>
  readonly save: (model: ServingModelId, defaults: ChatModelDefaults) => Effect.Effect<void, ChatModelDefaultsFailed>
  readonly clear: (model: ServingModelId) => Effect.Effect<void, ChatModelDefaultsFailed>
}
export class ChatModelDefaultsStore extends Context.Tag("ChatModelDefaultsStore")<
  ChatModelDefaultsStore,
  ChatModelDefaultsStoreApi
>() {}

const decodeDefaults = Schema.decodeUnknownOption(ChatModelDefaults)
const encodeDefaults = Schema.encodeSync(ChatModelDefaults)

/** The saved entries this version can read; an entry it can't is left in the file and skipped. */
export const readableModelDefaults = (entries: Readonly<Record<string, unknown>>): SavedChatModelDefaults =>
  Object.fromEntries(
    Object.entries(entries).flatMap(([model, value]) =>
      Option.match(decodeDefaults(value), {
        onNone: () => [],
        onSome: (defaults) => [[model, defaults] as const],
      })
    )
  )

/** Saved model defaults live in config.json beside the other server settings. */
export const ChatModelDefaultsStoreLive = Layer.effect(
  ChatModelDefaultsStore,
  Effect.gen(function* () {
    const fs = yield* FileSystem.FileSystem
    const host = yield* AcnHost
    const changes = yield* AcnChanges
    const storage = makeGlobalStorage({ root: host.dataDir })
    const config = yield* makeConfigStorage().pipe(Effect.provideService(GlobalStorage, storage))
    const unwritable = () => new ChatModelDefaultsFailed({ reason: "unwritable" })
    const update = (f: (saved: Readonly<Record<string, unknown>>) => Readonly<Record<string, unknown>>) =>
      config
        .update((current) => ({
          ...current,
          chatModelDefaults: Option.some(f(Option.getOrElse(current.chatModelDefaults, () => ({})))),
        }))
        .pipe(
          Effect.mapError(unwritable),
          Effect.zipRight(changes.publish({ operation: Chat.listModelDefaults._tag }))
        )

    return ChatModelDefaultsStore.of({
      list: readStructuredFile(storage.paths.configFile, MagnitudeConfigSchema.pick("chatModelDefaults")).pipe(
        Effect.provideService(FileSystem.FileSystem, fs),
        Effect.mapError(() => new ChatModelDefaultsFailed({ reason: "unreadable" })),
        Effect.flatMap((result) =>
          result._tag === "Invalid"
            ? Effect.fail(new ChatModelDefaultsFailed({ reason: "unreadable" }))
            : Effect.succeed(
                result._tag === "Missing"
                  ? {}
                  : Option.match(result.value.chatModelDefaults, {
                      onNone: () => ({}),
                      onSome: readableModelDefaults,
                    })
              )
        )
      ),
      save: (model, defaults) => update((saved) => ({ ...saved, [model]: encodeDefaults(defaults) })),
      clear: (model) =>
        update(({ [model]: _, ...rest }) => rest),
    })
  })
)
