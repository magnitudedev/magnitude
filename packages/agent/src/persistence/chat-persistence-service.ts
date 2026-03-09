/**
 * Chat Persistence Service
 *
 * Contract for persisting chat sessions to storage.
 * Implementations can use different backends (JSON files, SQLite, PostgreSQL, etc.)
 */

import { Effect, Context } from 'effect'
import type { AppEvent } from '../events'

// =============================================================================
// Error Types
// =============================================================================

export type PersistenceError =
  | { readonly _tag: 'LoadFailed'; readonly message: string }
  | { readonly _tag: 'SaveFailed'; readonly message: string }
  | { readonly _tag: 'BackendError'; readonly message: string }

export const PersistenceError = {
  LoadFailed: (message: string): PersistenceError => ({ _tag: 'LoadFailed', message }),
  SaveFailed: (message: string): PersistenceError => ({ _tag: 'SaveFailed', message }),
  BackendError: (message: string): PersistenceError => ({ _tag: 'BackendError', message })
}

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
}

export interface ChatPersistenceService {
  /**
   * Load all events for the current session
   */
  readonly loadEvents: () => Effect.Effect<AppEvent[], PersistenceError>

  /**
   * Persist new events (append-only)
   */
  readonly persistNewEvents: (events: AppEvent[]) => Effect.Effect<void, PersistenceError>

  /**
   * Get session metadata
   */
  readonly getSessionMetadata: () => Effect.Effect<SessionMetadata, PersistenceError>

  /**
   * Save mutable session metadata fields
   */
  readonly saveSessionMetadata: (
    update: Partial<Omit<SessionMetadata, 'sessionId' | 'created'>>
  ) => Effect.Effect<void, PersistenceError>
 
  readonly saveArtifact: (name: string, content: string) => Effect.Effect<void, PersistenceError>
}

export class ChatPersistence extends Context.Tag('ChatPersistence')<
  ChatPersistence,
  ChatPersistenceService
>() {}
