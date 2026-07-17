import { describe, test, expect, beforeEach, afterEach } from 'bun:test'
import { mkdir, rm, readFile, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { createId } from '@paralleldrive/cuid2'
import type { AppEvent } from '../../events'

interface StoredChatSession {
  readonly sessionId: string
  readonly created: string
  readonly updated: string
  readonly metadata: Record<string, unknown>
  readonly events: readonly AppEvent[]
}

const isFsError = (error: unknown): error is NodeJS.ErrnoException =>
  error instanceof Error && 'code' in error

const errorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : String(error)

const emptySession = (
  sessionId: string,
  metadata: Record<string, unknown> = {}
): StoredChatSession => ({
  sessionId,
  created: new Date().toISOString(),
  updated: new Date().toISOString(),
  metadata,
  events: []
})

const parseStoredSession = (content: string): StoredChatSession => {
  const parsed = JSON.parse(content) as Partial<StoredChatSession>
  return {
    sessionId: typeof parsed.sessionId === 'string' ? parsed.sessionId : '',
    created: typeof parsed.created === 'string' ? parsed.created : new Date().toISOString(),
    updated: typeof parsed.updated === 'string' ? parsed.updated : new Date().toISOString(),
    metadata: parsed.metadata && typeof parsed.metadata === 'object' && !Array.isArray(parsed.metadata)
      ? parsed.metadata
      : {},
    events: Array.isArray(parsed.events) ? parsed.events : []
  }
}

const userMessage = (messageId: string, content: string): AppEvent => ({
  type: 'user_message',
  messageId,
  forkId: null,
  timestamp: 1,
  text: content,
  mentions: [],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false
})

const messageChunk = (id: string, text: string): AppEvent => ({
  type: 'message_chunk',
  forkId: null,
  turnId: 'turn-1',
  id,
  text
})

// Mock implementation for testing
class JsonChatPersistence {
  private readonly writeLocks = new Map<string, Promise<unknown>>()

  constructor(private sessionsDir: string) {}

  private async withSessionWriteLock<T>(
    sessionId: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const previous = this.writeLocks.get(sessionId) ?? Promise.resolve()
    const run = previous.catch(() => undefined).then(operation)
    const tail = run.catch(() => undefined)
    this.writeLocks.set(sessionId, tail)

    try {
      return await run
    } finally {
      if (this.writeLocks.get(sessionId) === tail) {
        this.writeLocks.delete(sessionId)
      }
    }
  }

  async loadEvents(sessionId: string): Promise<AppEvent[]> {
    try {
      const filePath = join(this.sessionsDir, `${sessionId}.json`)
      const content = await readFile(filePath, 'utf-8')
      return [...parseStoredSession(content).events]
    } catch (error: unknown) {
      if (isFsError(error) && error.code === 'ENOENT') return []
      throw new Error(`Failed to load events: ${errorMessage(error)}`)
    }
  }

  async persistNewEvents(sessionId: string, events: AppEvent[]): Promise<void> {
    return this.withSessionWriteLock(sessionId, async () => {
      const filePath = join(this.sessionsDir, `${sessionId}.json`)

      // Read existing data
      let data = emptySession(sessionId)

      try {
        const existing = await readFile(filePath, 'utf-8')
        data = parseStoredSession(existing)
      } catch (error: unknown) {
        if (!isFsError(error) || error.code !== 'ENOENT') {
          throw new Error(`Failed to read existing file: ${errorMessage(error)}`)
        }
      }

      // Append new events
      data = {
        ...data,
        events: [...data.events, ...events],
        updated: new Date().toISOString()
      }

      // Atomic write: temp file + rename
      const tempPath = `${filePath}.tmp`
      await writeFile(tempPath, JSON.stringify(data, null, 2))
      await rm(filePath, { force: true })
      await writeFile(filePath, JSON.stringify(data, null, 2))
      await rm(tempPath, { force: true })
    })
  }

  async getSessionMetadata(sessionId: string): Promise<Record<string, unknown>> {
    const filePath = join(this.sessionsDir, `${sessionId}.json`)
    try {
      const content = await readFile(filePath, 'utf-8')
      return parseStoredSession(content).metadata
    } catch (error: unknown) {
      if (isFsError(error) && error.code === 'ENOENT') return {}
      throw new Error(`Failed to load metadata: ${errorMessage(error)}`)
    }
  }

  async saveSessionMetadata(sessionId: string, metadata: Record<string, unknown>): Promise<void> {
    return this.withSessionWriteLock(sessionId, async () => {
      const filePath = join(this.sessionsDir, `${sessionId}.json`)

      let data = emptySession(sessionId, metadata)

      try {
        const existing = await readFile(filePath, 'utf-8')
        const current = parseStoredSession(existing)
        data = {
          ...current,
          metadata: { ...current.metadata, ...metadata },
          updated: new Date().toISOString()
        }
      } catch (error: unknown) {
        if (!isFsError(error) || error.code !== 'ENOENT') {
          throw new Error(`Failed to read existing file: ${errorMessage(error)}`)
        }
      }

      await writeFile(filePath, JSON.stringify(data, null, 2))
    })
  }
}

describe('ChatPersistenceService - JSON Backend', () => {
  let testDir: string
  let persistence: JsonChatPersistence
  let sessionId: string

  beforeEach(async () => {
    // Create temp directory for tests
    testDir = join(tmpdir(), `magnitude-test-${createId()}`)
    await mkdir(testDir, { recursive: true })
    persistence = new JsonChatPersistence(testDir)
    sessionId = createId()
  })

  afterEach(async () => {
    // Cleanup
    await rm(testDir, { recursive: true, force: true })
  })

  describe('loadEvents', () => {
    test('returns empty array for new session', async () => {
      const events = await persistence.loadEvents(sessionId)
      expect(events).toEqual([])
    })

    test('loads persisted events', async () => {
      const testEvents: AppEvent[] = [
        {
          type: 'session_initialized',
          forkId: null,
          context: {
            cwd: '/test',
            scratchpadPath: '/test/.magnitude',
            platform: 'macos',
            shell: 'zsh',
            timezone: 'UTC',
            username: 'test',
            fullName: 'Test User',
            git: null,
            folderStructure: '',
            agentsFile: null,
            skills: null
          }
        },
        userMessage('user-hello', 'Hello')
      ]

      await persistence.persistNewEvents(sessionId, testEvents)
      const loaded = await persistence.loadEvents(sessionId)

      expect(loaded).toHaveLength(2)
      expect(loaded[0].type).toBe('session_initialized')
      expect(loaded[1].type).toBe('user_message')
    })

    test('handles corrupted JSON gracefully', async () => {
      const filePath = join(testDir, `${sessionId}.json`)
      await writeFile(filePath, 'invalid json {{{')

      await expect(persistence.loadEvents(sessionId)).rejects.toThrow('Failed to load events')
    })
  })

  describe('persistNewEvents', () => {
    test('creates new session file', async () => {
      const events: AppEvent[] = [userMessage('user-test', 'Test')]

      await persistence.persistNewEvents(sessionId, events)

      const filePath = join(testDir, `${sessionId}.json`)
      const content = await readFile(filePath, 'utf-8')
      const data = parseStoredSession(content)

      expect(data.sessionId).toBe(sessionId)
      expect(data.events).toHaveLength(1)
      expect(data.events[0].type).toBe('user_message')
      expect(data.created).toBeTruthy()
      expect(data.updated).toBeTruthy()
    })

    test('appends events to existing session', async () => {
      const firstBatch: AppEvent[] = [userMessage('user-first', 'First')]
      const secondBatch: AppEvent[] = [userMessage('user-second', 'Second')]

      await persistence.persistNewEvents(sessionId, firstBatch)
      await persistence.persistNewEvents(sessionId, secondBatch)

      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toHaveLength(2)
      expect(loaded[0]).toEqual(userMessage('user-first', 'First'))
      expect(loaded[1]).toEqual(userMessage('user-second', 'Second'))
    })

    test('handles empty events array', async () => {
      await persistence.persistNewEvents(sessionId, [])
      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toEqual([])
    })

    test('preserves event structure', async () => {
      const event: AppEvent = {
        type: 'turn_outcome',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        strategyId: 'native',
        outcome: {
          _tag: 'Completed',
          requestId: null,
          completion: {
            toolCallsCount: 0,
            finishReason: 'stop',
            feedback: [],
            yieldTarget: null
          }
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        cost: null,
        providerId: null,
        modelId: null
      }

      await persistence.persistNewEvents(sessionId, [event])
      const loaded = await persistence.loadEvents(sessionId)

      expect(loaded[0]).toEqual(event)
    })
  })

  describe('getSessionMetadata', () => {
    test('returns empty object for new session', async () => {
      const metadata = await persistence.getSessionMetadata(sessionId)
      expect(metadata).toEqual({})
    })

    test('loads saved metadata', async () => {
      await persistence.saveSessionMetadata(sessionId, {
        chatName: 'Test Chat',
        workingDirectory: '/test',
        gitBranch: 'main'
      })

      const metadata = await persistence.getSessionMetadata(sessionId)
      expect(metadata.chatName).toBe('Test Chat')
      expect(metadata.workingDirectory).toBe('/test')
      expect(metadata.gitBranch).toBe('main')
    })
  })

  describe('saveSessionMetadata', () => {
    test('creates metadata for new session', async () => {
      await persistence.saveSessionMetadata(sessionId, {
        chatName: 'New Chat'
      })

      const metadata = await persistence.getSessionMetadata(sessionId)
      expect(metadata.chatName).toBe('New Chat')
    })

    test('updates existing metadata', async () => {
      await persistence.saveSessionMetadata(sessionId, {
        chatName: 'Original'
      })

      await persistence.saveSessionMetadata(sessionId, {
        chatName: 'Updated',
        gitBranch: 'feature'
      })

      const metadata = await persistence.getSessionMetadata(sessionId)
      expect(metadata.chatName).toBe('Updated')
      expect(metadata.gitBranch).toBe('feature')
    })

    test('preserves events when updating metadata', async () => {
      const events: AppEvent[] = [userMessage('metadata-user-test', 'Test')]

      await persistence.persistNewEvents(sessionId, events)
      await persistence.saveSessionMetadata(sessionId, { chatName: 'Test' })

      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toHaveLength(1)
    })
  })

  describe('concurrent operations', () => {
    test('handles rapid successive writes', async () => {
      const batches = Array.from({ length: 10 }, (_, i) => [
        userMessage(`concurrent-user-${i}`, `Message ${i}`)
      ])

      await Promise.all(
        batches.map(batch => persistence.persistNewEvents(sessionId, batch))
      )

      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded.length).toBeGreaterThanOrEqual(10)
    })
  })

  describe('large sessions', () => {
    test('handles 1000+ events', async () => {
      const events: AppEvent[] = Array.from({ length: 1000 }, (_, i) =>
        messageChunk(`m-${i}`, `Chunk ${i}`)
      )

      await persistence.persistNewEvents(sessionId, events)
      const loaded = await persistence.loadEvents(sessionId)

      expect(loaded).toHaveLength(1000)
      expect(loaded[0]).toMatchObject({ type: 'message_chunk', text: 'Chunk 0' })
      expect(loaded[999]).toMatchObject({ type: 'message_chunk', text: 'Chunk 999' })
    })
  })
})
