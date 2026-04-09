import { Effect } from 'effect'
import * as path from 'path'
import type { StorageClient, StoredSessionMeta } from '@magnitudedev/storage'
import type { AppEvent, SessionMetadata, ChatPersistenceService } from '@magnitudedev/agent'
import { DEFAULT_CHAT_NAME, PersistenceError } from '@magnitudedev/agent'
import { textOf } from '@magnitudedev/agent'
import { CLI_VERSION } from '../version'

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

  private async readMetadata(): Promise<StoredSessionMeta | null> {
    return await this.storage.sessions.readMeta(this.sessionId)
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
              const next = this.updateSummaryFromEvents(
                current ?? this.buildMetadata(new Date().toISOString()),
                events
              )
              return {
                ...next,
                updated: new Date().toISOString(),
                lastActiveVersion: CLI_VERSION,
              }
            })
          } else {
            const now = new Date().toISOString()
            const newMetadata = this.updateSummaryFromEvents(this.buildMetadata(now), events)
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
        const metadata = await this.storage.sessions.readMeta(this.sessionId)

        if (!metadata) {
          const now = new Date().toISOString()
          return {
            sessionId: this.sessionId,
            chatName: DEFAULT_CHAT_NAME,
            workingDirectory: this.workingDirectory,
            gitBranch: null,
            created: now,
            updated: now,
            initialVersion: CLI_VERSION,
            lastActiveVersion: CLI_VERSION,
          }
        }

        return {
          sessionId: metadata.sessionId,
          chatName: metadata.chatName,
          workingDirectory: metadata.workingDirectory,
          gitBranch: metadata.gitBranch,
          created: metadata.created,
          updated: metadata.updated,
          initialVersion: metadata.initialVersion,
          lastActiveVersion: metadata.lastActiveVersion,
        }
      },
      catch: (error) => new PersistenceError({ reason: 'LoadFailed', message: error instanceof Error ? error.message : String(error) })
    })

  readonly saveSessionMetadata = (
    update: Partial<Omit<SessionMetadata, 'sessionId' | 'created' | 'initialVersion' | 'lastActiveVersion'>>
  ): Effect.Effect<void, PersistenceError> =>
    Effect.tryPromise({
      try: async () => {
        await this.queueWrite(async () => {
          await this.storage.sessions.updateMeta(this.sessionId, (metadata) => ({
            ...(metadata ?? this.buildMetadata(new Date().toISOString())),
            ...update,
            updated: new Date().toISOString(),
            lastActiveVersion: CLI_VERSION,
          }))
        })
      },
      catch: (error) => new PersistenceError({ reason: 'SaveFailed', message: error instanceof Error ? error.message : String(error) })
    })

  /**
   * Update summary fields on metadata from new events
   */
  private buildMetadata(now: string): StoredSessionMeta {
    return {
      sessionId: this.sessionId,
      created: now,
      updated: now,
      chatName: DEFAULT_CHAT_NAME,
      workingDirectory: this.workingDirectory,
      initialVersion: CLI_VERSION,
      lastActiveVersion: CLI_VERSION,
      gitBranch: null,
      firstUserMessage: null,
      lastMessage: null,
      messageCount: 0,
    }
  }

  private updateSummaryFromEvents(metadata: StoredSessionMeta, events: AppEvent[]): StoredSessionMeta {
    let next = metadata

    for (const event of events) {
      const e = event as any
      if (e.type === 'user_message') {
        const text = textOf(e.content).trim()
        if (!text) continue
        next = {
          ...next,
          messageCount: next.messageCount + 1,
          firstUserMessage: next.firstUserMessage ?? text,
          lastMessage: text,
        }
        continue
      }
      if (e.type === 'text_chunk' && typeof e.text === 'string') {
        const text = e.text.trim()
        if (text) {
          next = {
            ...next,
            lastMessage: text,
          }
        }
      }
    }

    return next
  }

  getSessionId(): string {
    return this.sessionId
  }

  getSessionDir(): string {
    return path.dirname(this.storage.sessions.getEventsPath(this.sessionId))
  }
}
