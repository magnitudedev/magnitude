import { Effect } from 'effect'
import * as path from 'path'
import type { StorageClient } from '@magnitudedev/storage'
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

export class JsonChatPersistence implements ChatPersistenceService {
  private sessionId: string
  private storage: StorageClient
  private workingDirectory: string
  private writeQueue: Promise<unknown> = Promise.resolve()

  constructor(options: {
    storage: StorageClient
    workingDirectory: string
    sessionId?: string
  }) {
    this.storage = options.storage
    this.workingDirectory = options.workingDirectory
    this.sessionId = options.sessionId ?? options.storage.sessions.createId()
  }

  private async readMetadata(): Promise<MetadataFile | null> {
    return await this.storage.sessions.readMeta(this.sessionId) as MetadataFile | null
  }

  /**
   * Queue a write operation to prevent concurrent write conflicts
   */
  private queueWrite<T>(operation: () => Promise<T>): Promise<T> {
    const queued = this.writeQueue.then(() => operation())
    this.writeQueue = queued.catch(() => {})
    return queued
  }

  private async readEvents(): Promise<AppEvent[]> {
    return await this.storage.sessions.readEvents<AppEvent>(this.sessionId)
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
            await this.storage.sessions.appendEvents(this.sessionId, events)
            await this.storage.sessions.updateMeta(this.sessionId, (current) => {
              const next = { ...(current as MetadataFile) }
              this.updateSummaryFromEvents(next, events)
              next.updated = new Date().toISOString()
              return next
            })
          } else {
            const now = new Date().toISOString()
            const newMetadata: MetadataFile = {
              sessionId: this.sessionId,
              created: now,
              updated: now,
              chatName: DEFAULT_CHAT_NAME,
              workingDirectory: this.workingDirectory,
              gitBranch: null,
              firstUserMessage: null,
              lastMessage: null,
              messageCount: 0
            }
            this.updateSummaryFromEvents(newMetadata, events)
            await this.storage.sessions.writeMeta(this.sessionId, newMetadata)
            await this.storage.sessions.appendEvents(this.sessionId, events)
          }
        })
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly getSessionMetadata = (): Effect.Effect<SessionMetadata, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        const metadata = await this.storage.sessions.readMeta(this.sessionId) as MetadataFile | null

        if (!metadata) {
          const now = new Date().toISOString()
          return {
            sessionId: this.sessionId,
            chatName: DEFAULT_CHAT_NAME,
            workingDirectory: this.workingDirectory,
            gitBranch: null,
            created: now,
            updated: now
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
          await this.storage.sessions.updateMeta(this.sessionId, (metadata) => ({
            ...(metadata as MetadataFile),
            ...update,
            updated: new Date().toISOString(),
          }))
        })
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: error instanceof Error ? error.message : String(error) })
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
    return path.dirname(this.storage.sessions.getEventsPath(this.sessionId))
  }
}
