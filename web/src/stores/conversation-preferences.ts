import { Atom, useAtomMount, useAtomSet, useAtomValue } from "@effect-atom/atom-react"
import { Data, Effect, Either, Schema } from "effect"
import { useCallback, useMemo } from "react"
import { usePlatform, type Storage } from "@magnitudedev/client-common"

const STORAGE_KEY = "conversation-preferences"

export const ConversationPreferencesSchema = Schema.Struct({
  showThinking: Schema.Boolean,
})
export type ConversationPreferences = typeof ConversationPreferencesSchema.Type

const StoredConversationPreferencesSchema = Schema.parseJson(ConversationPreferencesSchema)

class ConversationPreferenceStorageError extends Data.TaggedError(
  "ConversationPreferenceStorageError"
)<{
  readonly operation: "read" | "write"
}> {}

export const defaultConversationPreferences: ConversationPreferences = {
  showThinking: false,
}

export function decodeConversationPreferences(
  value: string | null,
): ConversationPreferences {
  if (value === null) return defaultConversationPreferences
  const decoded = Schema.decodeUnknownEither(StoredConversationPreferencesSchema)(value)
  return Either.isRight(decoded) ? decoded.right : defaultConversationPreferences
}

export function encodeConversationPreferences(
  value: ConversationPreferences,
): string {
  return Schema.encodeSync(StoredConversationPreferencesSchema)(value)
}

interface ConversationPreferenceState {
  readonly preferences: ConversationPreferences
  readonly hydrated: boolean
}

const conversationPreferencesAtom = Atom.keepAlive(
  Atom.make<ConversationPreferenceState>({
    preferences: defaultConversationPreferences,
    hydrated: false,
  }),
)

function readPreferences(storage: Storage) {
  return Effect.tryPromise({
    try: () => storage.getItem(STORAGE_KEY),
    catch: () => new ConversationPreferenceStorageError({ operation: "read" }),
  }).pipe(
    Effect.map(decodeConversationPreferences),
  )
}

function writePreferences(storage: Storage, preferences: ConversationPreferences) {
  return Effect.tryPromise({
    try: () => storage.setItem(STORAGE_KEY, encodeConversationPreferences(preferences)),
    catch: () => new ConversationPreferenceStorageError({ operation: "write" }),
  })
}

export function useInitializeConversationPreferences(): void {
  const platform = usePlatform()
  const setPreferences = useAtomSet(conversationPreferencesAtom)
  const initializationAtom = useMemo(
    () => Atom.make(readPreferences(platform.storage).pipe(
      Effect.tap((preferences) => Effect.sync(() => setPreferences((current) =>
        current.hydrated ? current : { preferences, hydrated: true }
      ))),
      Effect.catchTag("ConversationPreferenceStorageError", (error) =>
        Effect.logWarning("Could not load conversation preferences").pipe(
          Effect.annotateLogs({ operation: error.operation }),
        )),
    )),
    [platform.storage, setPreferences],
  )
  useAtomMount(initializationAtom)
}

export function useShowThinkingPreference(): readonly [boolean, (value: boolean) => void] {
  const platform = usePlatform()
  const state = useAtomValue(conversationPreferencesAtom)
  const setState = useAtomSet(conversationPreferencesAtom)
  const update = useCallback((showThinking: boolean) => {
    const next = { ...state.preferences, showThinking }
    setState({ preferences: next, hydrated: true })
    Effect.runFork(writePreferences(platform.storage, next).pipe(
      Effect.catchTag("ConversationPreferenceStorageError", (error) =>
        Effect.logWarning("Could not save conversation preferences").pipe(
          Effect.annotateLogs({ operation: error.operation }),
        )),
    ))
  }, [platform.storage, state.preferences, setState])
  return [state.preferences.showThinking, update]
}

export function useShowThinking(): boolean {
  return useAtomValue(conversationPreferencesAtom).preferences.showThinking
}
