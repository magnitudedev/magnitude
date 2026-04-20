import { Context, Effect, Layer, Ref } from 'effect'
import type { AppEvent } from '../events'
import {
  ChatPersistence,
  type ChatPersistenceService,
  type SessionMetadata,
} from '../persistence/chat-persistence-service'

export interface InMemoryPersistenceSeed {
  readonly events?: readonly AppEvent[]
  readonly artifacts?: Readonly<Record<string, string>>
  readonly metadata?: Partial<SessionMetadata>
}

export interface InMemoryPersistenceState {
  readonly events: readonly AppEvent[]
  readonly artifacts: Readonly<Record<string, string>>
  readonly metadata: SessionMetadata
}

export interface InMemoryChatPersistence extends ChatPersistenceService {
  readonly inspectEvents: () => Effect.Effect<readonly AppEvent[]>
  readonly inspectArtifacts: () => Effect.Effect<Readonly<Record<string, string>>>
  readonly inspectMetadata: () => Effect.Effect<SessionMetadata>
  readonly inspectState: () => Effect.Effect<InMemoryPersistenceState>
}

interface InternalState {
  readonly events: AppEvent[]
  readonly artifacts: Record<string, string>
  readonly metadata: SessionMetadata
}

const nowIso = () => new Date().toISOString()

const buildInitialMetadata = (seed?: Partial<SessionMetadata>): SessionMetadata => {
  const now = nowIso()
  return {
    sessionId: seed?.sessionId ?? 'test-session',
    chatName: seed?.chatName ?? 'Test Chat',
    workingDirectory: seed?.workingDirectory ?? process.cwd(),
    gitBranch: seed?.gitBranch ?? null,
    created: seed?.created ?? now,
    updated: seed?.updated ?? now,
    initialVersion: seed?.initialVersion ?? '0.0.1',
    lastActiveVersion: seed?.lastActiveVersion ?? '0.0.1',
  }
}

const snapshot = (state: InternalState): InMemoryPersistenceState => ({
  events: [...state.events],
  artifacts: { ...state.artifacts },
  metadata: { ...state.metadata },
})

const makeService = (stateRef: Ref.Ref<InternalState>): InMemoryChatPersistence => ({
  loadEvents: () =>
    Ref.get(stateRef).pipe(Effect.map((state) => [...state.events])),

  persistNewEvents: (events) =>
    Ref.update(stateRef, (state) => ({
      ...state,
      events: [...state.events, ...events],
      metadata: {
        ...state.metadata,
        updated: nowIso(),
      },
    })),

  getSessionMetadata: () =>
    Ref.get(stateRef).pipe(Effect.map((state) => ({ ...state.metadata }))),

  saveSessionMetadata: (update) =>
    Ref.update(stateRef, (state) => ({
      ...state,
      metadata: {
        ...state.metadata,
        ...update,
        sessionId: state.metadata.sessionId,
        created: state.metadata.created,
        updated: nowIso(),
      },
    })),

  inspectEvents: () =>
    Ref.get(stateRef).pipe(Effect.map((state) => [...state.events])),

  inspectArtifacts: () =>
    Ref.get(stateRef).pipe(Effect.map((state) => ({ ...state.artifacts }))),

  inspectMetadata: () =>
    Ref.get(stateRef).pipe(Effect.map((state) => ({ ...state.metadata }))),

  inspectState: () =>
    Ref.get(stateRef).pipe(Effect.map(snapshot)),
})

export const makeInMemoryChatPersistence = (
  seed: InMemoryPersistenceSeed = {}
): Effect.Effect<InMemoryChatPersistence, never> =>
  Effect.gen(function* () {
    const stateRef = yield* Ref.make<InternalState>({
      events: [...(seed.events ?? [])],
      artifacts: { ...(seed.artifacts ?? {}) },
      metadata: buildInitialMetadata(seed.metadata),
    })
    return makeService(stateRef)
  })

export const InMemoryChatPersistenceTag = Context.GenericTag<InMemoryChatPersistence>(
  '@magnitudedev/agent/test-harness/InMemoryChatPersistence'
)

export const makeInMemoryChatPersistenceLayer = (
  seed: InMemoryPersistenceSeed = {}
): Layer.Layer<ChatPersistence | InMemoryChatPersistence, never, never> => {
  const inMemoryLayer = Layer.effect(
    InMemoryChatPersistenceTag,
    makeInMemoryChatPersistence(seed)
  )

  const chatPersistenceLayer = Layer.effect(
    ChatPersistence,
    Effect.map(InMemoryChatPersistenceTag, (service): ChatPersistenceService => service)
  ).pipe(Layer.provide(inMemoryLayer))

  return Layer.merge(inMemoryLayer, chatPersistenceLayer)
}