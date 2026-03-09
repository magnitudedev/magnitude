/**
 * In-Memory Persistence Implementation
 *
 * Test implementation that stores everything in memory.
 * Useful for testing without file I/O.
 */

import type { AppEvent } from '@magnitudedev/agent'
import type { ChatPersistenceService, SessionMetadata } from './chat-persistence'

export class InMemoryPersistence implements ChatPersistenceService {
  private metadata: SessionMetadata
  private events: AppEvent[] = []

  constructor(sessionId: string, workingDirectory: string, gitBranch: string | null) {
    this.metadata = {
      sessionId,
      chatName: null,
      workingDirectory,
      gitBranch,
      created: new Date().toISOString(),
      updated: new Date().toISOString()
    }
  }

  async exists(): Promise<boolean> {
    return this.events.length > 0
  }

  async getSessionMetadata(): Promise<SessionMetadata> {
    return { ...this.metadata }
  }

  async saveSessionMetadata(update: Partial<Pick<SessionMetadata, 'chatName'>>): Promise<void> {
    this.metadata = {
      ...this.metadata,
      updated: new Date().toISOString(),
      ...(update.chatName !== undefined && { chatName: update.chatName })
    }
  }

  async persistNewEvents(events: AppEvent[]): Promise<void> {
    this.events.push(...events)
    this.metadata = {
      ...this.metadata,
      updated: new Date().toISOString()
    }
  }

  async loadEvents(): Promise<AppEvent[]> {
    return [...this.events]
  }

  // Test helpers
  clear(): void {
    this.events = []
  }

  getEventCount(): number {
    return this.events.length
  }
}

export function createInMemoryPersistence(
  sessionId: string,
  workingDirectory: string,
  gitBranch: string | null
): ChatPersistenceService {
  return new InMemoryPersistence(sessionId, workingDirectory, gitBranch)
}
