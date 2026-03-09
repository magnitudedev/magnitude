/**
 * ChatPersistenceService
 *
 * Service for persisting chat session state to JSON files.
 * Adapted from Sage's architecture for Magnitude's needs.
 */

import type { AppEvent } from '@magnitudedev/agent'

// ============================================================================
// Error Types
// ============================================================================

export type PersistenceError =
  | { type: 'load_failed'; message: string }
  | { type: 'save_failed'; message: string }
  | { type: 'file_error'; message: string }

// ============================================================================
// Session Metadata
// ============================================================================

export interface SessionMetadata {
  readonly sessionId: string
  readonly chatName: string | null
  readonly workingDirectory: string
  readonly gitBranch: string | null
  readonly created: string  // ISO timestamp
  readonly updated: string  // ISO timestamp
}

// ============================================================================
// Service Interface
// ============================================================================

export interface ChatPersistenceService {
  /**
   * Get session metadata (name, timestamps, context).
   */
  getSessionMetadata: () => Promise<SessionMetadata>

  /**
   * Update session metadata (e.g., chat name).
   */
  saveSessionMetadata: (update: Partial<Pick<SessionMetadata, 'chatName'>>) => Promise<void>

  /**
   * Persist new events to storage (append-only).
   * Called by LifecycleCoordinator when pending events need to be saved.
   */
  persistNewEvents: (events: AppEvent[]) => Promise<void>

  /**
   * Load all events for this session from storage.
   * Returns events in order they were persisted.
   */
  loadEvents: () => Promise<AppEvent[]>

  /**
   * Check if session file exists.
   */
  exists: () => Promise<boolean>
}
