/**
 * JSON File Persistence Implementation
 *
 * Stores chat sessions as JSON files in ~/.magnitude/sessions/
 * Uses atomic write pattern (temp file + rename) for safety.
 */

import { mkdir, readFile, writeFile, rename, access } from 'node:fs/promises'
import { constants } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import type { AppEvent } from '@magnitudedev/agent'
import type { ChatPersistenceService, SessionMetadata, PersistenceError } from './chat-persistence'

// ============================================================================
// Session File Format
// ============================================================================

interface SessionFile {
  readonly sessionId: string
  readonly created: string
  readonly updated: string
  readonly metadata: {
    readonly chatName: string | null
    readonly workingDirectory: string
    readonly gitBranch: string | null
  }
  readonly events: AppEvent[]
}

// ============================================================================
// Implementation
// ============================================================================

export class JsonFilePersistence implements ChatPersistenceService {
  private readonly sessionId: string
  private readonly sessionDir: string
  private readonly sessionPath: string
  private cachedMetadata: SessionMetadata | null = null

  constructor(sessionId: string, workingDirectory: string, gitBranch: string | null) {
    this.sessionId = sessionId
    this.sessionDir = join(homedir(), '.magnitude', 'sessions')
    this.sessionPath = join(this.sessionDir, `${sessionId}.json`)
    
    // Initialize metadata cache for new sessions
    this.cachedMetadata = {
      sessionId,
      chatName: null,
      workingDirectory,
      gitBranch,
      created: new Date().toISOString(),
      updated: new Date().toISOString()
    }
  }

  async exists(): Promise<boolean> {
    try {
      await access(this.sessionPath, constants.F_OK)
      return true
    } catch {
      return false
    }
  }

  async getSessionMetadata(): Promise<SessionMetadata> {
    // Return cached metadata if available
    if (this.cachedMetadata) {
      return this.cachedMetadata
    }

    try {
      const sessionFile = await this.readSessionFile()
      this.cachedMetadata = {
        sessionId: sessionFile.sessionId,
        chatName: sessionFile.metadata.chatName,
        workingDirectory: sessionFile.metadata.workingDirectory,
        gitBranch: sessionFile.metadata.gitBranch,
        created: sessionFile.created,
        updated: sessionFile.updated
      }
      return this.cachedMetadata
    } catch (error) {
      throw this.toError('load_failed', error)
    }
  }

  async saveSessionMetadata(update: Partial<Pick<SessionMetadata, 'chatName'>>): Promise<void> {
    try {
      const sessionFile = await this.readSessionFile()
      const updated: SessionFile = {
        ...sessionFile,
        updated: new Date().toISOString(),
        metadata: {
          ...sessionFile.metadata,
          ...(update.chatName !== undefined && { chatName: update.chatName })
        }
      }
      
      await this.writeSessionFile(updated)
      
      // Update cache
      if (this.cachedMetadata) {
        this.cachedMetadata = {
          ...this.cachedMetadata,
          updated: updated.updated,
          ...(update.chatName !== undefined && { chatName: update.chatName })
        }
      }
    } catch (error) {
      throw this.toError('save_failed', error)
    }
  }

  async persistNewEvents(events: AppEvent[]): Promise<void> {
    if (events.length === 0) return

    try {
      // Ensure directory exists
      await mkdir(this.sessionDir, { recursive: true })

      let sessionFile: SessionFile
      
      if (await this.exists()) {
        // Load existing file and append events
        sessionFile = await this.readSessionFile()
        sessionFile = {
          ...sessionFile,
          updated: new Date().toISOString(),
          events: [...sessionFile.events, ...events]
        }
      } else {
        // Create new session file
        const metadata = await this.getSessionMetadata()
        sessionFile = {
          sessionId: this.sessionId,
          created: metadata.created,
          updated: new Date().toISOString(),
          metadata: {
            chatName: metadata.chatName,
            workingDirectory: metadata.workingDirectory,
            gitBranch: metadata.gitBranch
          },
          events
        }
      }

      await this.writeSessionFile(sessionFile)
    } catch (error) {
      throw this.toError('save_failed', error)
    }
  }

  async loadEvents(): Promise<AppEvent[]> {
    try {
      if (!(await this.exists())) {
        return []
      }

      const sessionFile = await this.readSessionFile()
      return sessionFile.events
    } catch (error) {
      throw this.toError('load_failed', error)
    }
  }

  // ============================================================================
  // Private Helpers
  // ============================================================================

  private async readSessionFile(): Promise<SessionFile> {
    const content = await readFile(this.sessionPath, 'utf-8')
    return JSON.parse(content) as SessionFile
  }

  private async writeSessionFile(data: SessionFile): Promise<void> {
    // Atomic write: write to temp file, then rename
    const tempPath = `${this.sessionPath}.tmp`
    await writeFile(tempPath, JSON.stringify(data, null, 2), 'utf-8')
    await rename(tempPath, this.sessionPath)
  }

  private toError(type: PersistenceError['type'], error: unknown): PersistenceError {
    const message = error instanceof Error ? error.message : String(error)
    return { type, message } as PersistenceError
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export function createJsonFilePersistence(
  sessionId: string,
  workingDirectory: string,
  gitBranch: string | null
): ChatPersistenceService {
  return new JsonFilePersistence(sessionId, workingDirectory, gitBranch)
}
