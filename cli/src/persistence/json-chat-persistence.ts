import { Effect } from 'effect'
import * as fs from 'fs/promises'
import * as path from 'path'
import * as os from 'os'
import type { AppEvent, SessionMetadata, ChatPersistenceService } from '@magnitudedev/agent'
import { DEFAULT_CHAT_NAME, PersistenceError } from '@magnitudedev/agent'
import { textOf } from '@magnitudedev/agent'

interface MetadataFile {
  sessionId: string
  created: string
  updated: string
  chatName: string
  workingDirectory: string
  gitBranch: string | null
  firstUserMessage: string | null
  lastMessage: string | null
  messageCount: number
}

/**
 * JSON file-based persistence implementation using UTC timestamp folders
 * 
 * Structure:
 * ~/.magnitude/sessions/
 *   2026-02-10T22-01-18Z/
 *     events.jsonl
 *     meta.json
 *     logs.jsonl
 */
export class JsonChatPersistence implements ChatPersistenceService {
  private sessionId: string
  private sessionDir: string
  private eventsPath: string
  private metaPath: string
  private writeQueue: Promise<unknown> = Promise.resolve()

  constructor(sessionId?: string) {
    this.sessionId = sessionId ?? this.createTimestampSessionId()
    const sessionsDir = path.join(os.homedir(), '.magnitude', 'sessions')
    this.sessionDir = path.join(sessionsDir, this.sessionId)
    this.eventsPath = path.join(this.sessionDir, 'events.jsonl')
    this.metaPath = path.join(this.sessionDir, 'meta.json')
  }

  /**
   * Create a session ID from current UTC timestamp
   * Format: YYYY-MM-DDTHH-MM-SSZ (colons replaced with hyphens for filesystem compatibility)
   */
  private createTimestampSessionId(): string {
    const now = new Date()
    return now.toISOString()
      .replace(/:/g, '-')
      .replace(/\.\d{3}Z$/, 'Z')
  }

  /**
   * Find the latest session by scanning timestamp folders
   */
  static async findLatestSessionId(): Promise<string | null> {
    const sessionsDir = path.join(os.homedir(), '.magnitude', 'sessions')
    try {
      const entries = await fs.readdir(sessionsDir, { withFileTypes: true })
      const sessionDirs = entries
        .filter(entry => entry.isDirectory())
        .map(entry => entry.name)
        .filter(name => /^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z$/.test(name))
        .sort()
        .reverse()

      return sessionDirs[0] ?? null
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') {
        return null
      }
      throw error
    }
  }

  private async ensureSessionDir(): Promise<void> {
    await fs.mkdir(this.sessionDir, { recursive: true })
  }

  private async readMetadata(): Promise<MetadataFile | null> {
    const file = Bun.file(this.metaPath)
    if (!(await file.exists())) return null
    
    return await file.json() as MetadataFile
  }

  private async writeMetadata(data: MetadataFile): Promise<void> {
    await this.ensureSessionDir()
    await Bun.write(this.metaPath, JSON.stringify(data, null, 2))
  }

  private async readEvents(): Promise<AppEvent[]> {
    const file = Bun.file(this.eventsPath)
    if (!(await file.exists())) return []
    
    const content = await file.text()
    const lines = content.trim().split('\n').filter(line => line.length > 0)
    return lines.map(line => JSON.parse(line) as AppEvent)
  }

  /**
   * Queue a write operation to prevent concurrent write conflicts
   */
  private queueWrite<T>(operation: () => Promise<T>): Promise<T> {
    const queued = this.writeQueue.then(() => operation())
    this.writeQueue = queued.catch(() => {})
    return queued
  }

  private async appendEvents(events: AppEvent[]): Promise<void> {
    await this.ensureSessionDir()
    
    const lines = events.map(event => JSON.stringify(event)).join('\n') + '\n'
    
    // Append-only write — O(new data) instead of O(total file size)
    await fs.appendFile(this.eventsPath, lines)
  }

  readonly loadEvents = (): Effect.Effect<AppEvent[], PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        return await this.readEvents()
      },
      catch: (error) => new PersistenceError({ reason: 'LoadFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly persistNewEvents = (events: AppEvent[]): Effect.Effect<void, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        await this.queueWrite(async () => {
          const metadata = await this.readMetadata()
          
          if (metadata) {
            // Update summary fields from new events
            this.updateSummaryFromEvents(metadata, events)
            metadata.updated = new Date().toISOString()
            await this.writeMetadata(metadata)
            await this.appendEvents(events)
          } else {
            // Create new session — write metadata once on first persist
            const newMetadata: MetadataFile = {
              sessionId: this.sessionId,
              created: new Date().toISOString(),
              updated: new Date().toISOString(),
              chatName: DEFAULT_CHAT_NAME,
              workingDirectory: process.cwd(),
              gitBranch: null,
              firstUserMessage: null,
              lastMessage: null,
              messageCount: 0
            }
            this.updateSummaryFromEvents(newMetadata, events)
            await this.writeMetadata(newMetadata)
            await this.appendEvents(events)
          }
        })
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly getSessionMetadata = (): Effect.Effect<SessionMetadata, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        const metadata = await this.readMetadata()
        
        if (!metadata) {
          return {
            sessionId: this.sessionId,
            chatName: DEFAULT_CHAT_NAME,
            workingDirectory: process.cwd(),
            gitBranch: null,
            created: new Date().toISOString(),
            updated: new Date().toISOString()
          }
        }

        return {
          sessionId: metadata.sessionId,
          chatName: metadata.chatName,
          workingDirectory: metadata.workingDirectory,
          gitBranch: metadata.gitBranch,
          created: metadata.created,
          updated: metadata.updated
        }
      },
      catch: (error) => new PersistenceError({ reason: 'LoadFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly saveSessionMetadata = (
    update: Partial<Omit<SessionMetadata, 'sessionId' | 'created'>>
  ): Effect.Effect<void, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        await this.queueWrite(async () => {
          const metadata = await this.readMetadata()
          
          if (!metadata) {
            throw new Error('Cannot update metadata: session does not exist')
          }

          if (update.chatName !== undefined) {
            metadata.chatName = update.chatName
          }
          if (update.workingDirectory !== undefined) {
            metadata.workingDirectory = update.workingDirectory
          }
          if (update.gitBranch !== undefined) {
            metadata.gitBranch = update.gitBranch
          }
          
          metadata.updated = new Date().toISOString()
          await this.writeMetadata(metadata)
        })
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly saveArtifact = (name: string, content: string): Effect.Effect<void, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        const artifactsDir = path.join(this.sessionDir, 'artifacts')
        await fs.mkdir(artifactsDir, { recursive: true })
        await fs.writeFile(path.join(artifactsDir, name + '.md'), content)
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: `Failed to save artifact "${name}": ${error}` })
    })

  /**
   * Update summary fields on metadata from new events
   */
  private updateSummaryFromEvents(metadata: MetadataFile, events: AppEvent[]): void {

    for (const event of events) {
      const e = event as any
      if (e.type === 'user_message') {
        const text = textOf(e.content).trim()
        if (!text) continue
        metadata.messageCount += 1
        if (!metadata.firstUserMessage) metadata.firstUserMessage = text
        metadata.lastMessage = text
        continue
      }
      if (e.type === 'text_chunk' && typeof e.text === 'string') {
        const text = e.text.trim()
        if (text) metadata.lastMessage = text
      }
    }
  }

  getSessionId(): string {
    return this.sessionId
  }

  getSessionDir(): string {
    return this.sessionDir
  }
}
