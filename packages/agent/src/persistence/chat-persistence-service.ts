/**
 * Chat Persistence Service
 *
 * Contract for persisting chat sessions to storage.
 * Implementations can use different backends (JSON files, SQLite, PostgreSQL, etc.)
 */

import { Effect, Context, Data, Schema } from 'effect'
import type { EventCursor, Timestamped } from '@magnitudedev/event-core'
import type { AppEvent } from '../events'

// =============================================================================
// Error Types
// =============================================================================

export class PersistenceError extends Data.TaggedError('PersistenceError')<{
  readonly reason: 'LoadFailed' | 'SaveFailed' | 'BackendError'
  readonly message: string
}> {}

// =============================================================================
// Service Interface
// =============================================================================

export interface SessionMetadata {
  readonly sessionId: string
  readonly chatName: string
  readonly workingDirectory: string
  readonly gitBranch: string | null
  readonly created: string
  readonly updated: string
  readonly initialVersion: string
  readonly lastActiveVersion: string
}

export interface ChatPersistenceService {
  readonly loadEvents: () => Effect.Effect<Timestamped<AppEvent>[], PersistenceError>
  readonly loadEventsAfterCursor: (cursor: EventCursor) => Effect.Effect<Timestamped<AppEvent>[] | null, PersistenceError>
  readonly persistNewEvents: (events: Timestamped<AppEvent>[]) => Effect.Effect<EventCursor | null, PersistenceError>
  readonly loadProjectionSnapshot: () => Effect.Effect<unknown | null, PersistenceError>
  readonly saveProjectionSnapshot: <A>(snapshot: A) => Effect.Effect<void, PersistenceError>
  readonly getSessionMetadata: () => Effect.Effect<SessionMetadata, PersistenceError>
  readonly saveSessionMetadata: (
    update: Partial<Omit<SessionMetadata, 'sessionId' | 'created' | 'initialVersion' | 'lastActiveVersion'>>
  ) => Effect.Effect<void, PersistenceError>
}

export class ChatPersistence extends Context.Tag('ChatPersistence')<
  ChatPersistence,
  ChatPersistenceService
>() {}
