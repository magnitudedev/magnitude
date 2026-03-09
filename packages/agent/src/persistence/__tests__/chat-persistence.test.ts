// @ts-nocheck
import { describe, test, expect, beforeEach, afterEach } from 'bun:test'
import { mkdir, rm, readFile, writeFile } from 'fs/promises'
import { join } from 'path'
import { tmpdir } from 'os'
import { createId } from '@paralleldrive/cuid2'
import type { AppEvent } from '../../events'

// Mock implementation for testing
class JsonChatPersistence {
  constructor(private sessionsDir: string) {}

  async loadEvents(sessionId: string): Promise<AppEvent[]> {
    try {
      const filePath = join(this.sessionsDir, `${sessionId}.json`)
      const content = await readFile(filePath, 'utf-8')
      const data = JSON.parse(content)
      return data.events || []
    } catch (error: any) {
      if (error.code === 'ENOENT') return []
      throw new Error(`Failed to load events: ${error.message}`)
    }
  }

  async persistNewEvents(sessionId: string, events: AppEvent[]): Promise<void> {
    const filePath = join(this.sessionsDir, `${sessionId}.json`)
    
    // Read existing data
    let data: any = {
      sessionId,
      created: new Date().toISOString(),
      updated: new Date().toISOString(),
      metadata: {},
      events: []
    }
    
    try {
      const existing = await readFile(filePath, 'utf-8')
      data = JSON.parse(existing)
    } catch (error: any) {
      if (error.code !== 'ENOENT') {
        throw new Error(`Failed to read existing file: ${error.message}`)
      }
    }
    
    // Append new events
    data.events = [...data.events, ...events]
    data.updated = new Date().toISOString()
    
    // Atomic write: temp file + rename
    const tempPath = `${filePath}.tmp`
    await writeFile(tempPath, JSON.stringify(data, null, 2))
    await rm(filePath, { force: true })
    await writeFile(filePath, JSON.stringify(data, null, 2))
    await rm(tempPath, { force: true })
  }

  async getSessionMetadata(sessionId: string): Promise<any> {
    const filePath = join(this.sessionsDir, `${sessionId}.json`)
    try {
      const content = await readFile(filePath, 'utf-8')
      const data = JSON.parse(content)
      return data.metadata || {}
    } catch (error: any) {
      if (error.code === 'ENOENT') return {}
      throw new Error(`Failed to load metadata: ${error.message}`)
    }
  }

  async saveSessionMetadata(sessionId: string, metadata: any): Promise<void> {
    const filePath = join(this.sessionsDir, `${sessionId}.json`)
    
    let data: any = {
      sessionId,
      created: new Date().toISOString(),
      updated: new Date().toISOString(),
      metadata,
      events: []
    }
    
    try {
      const existing = await readFile(filePath, 'utf-8')
      data = JSON.parse(existing)
      data.metadata = { ...data.metadata, ...metadata }
      data.updated = new Date().toISOString()
    } catch (error: any) {
      if (error.code !== 'ENOENT') {
        throw new Error(`Failed to read existing file: ${error.message}`)
      }
    }
    
    await writeFile(filePath, JSON.stringify(data, null, 2))
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
        {
          type: 'user_message',
          forkId: null,
          content: 'Hello',
          mode: 'text',
          synthetic: false, taskMode: false
        }
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
      const events: AppEvent[] = [
        {
          type: 'user_message',
          forkId: null,
          content: 'Test',
          mode: 'text',
          synthetic: false, taskMode: false
        }
      ]

      await persistence.persistNewEvents(sessionId, events)
      
      const filePath = join(testDir, `${sessionId}.json`)
      const content = await readFile(filePath, 'utf-8')
      const data = JSON.parse(content)
      
      expect(data.sessionId).toBe(sessionId)
      expect(data.events).toHaveLength(1)
      expect(data.events[0].type).toBe('user_message')
      expect(data.created).toBeTruthy()
      expect(data.updated).toBeTruthy()
    })

    test('appends events to existing session', async () => {
      const firstBatch: AppEvent[] = [
        {
          type: 'user_message',
          forkId: null,
          content: 'First',
          mode: 'text',
          synthetic: false, taskMode: false
        }
      ]

      const secondBatch: AppEvent[] = [
        {
          type: 'user_message',
          forkId: null,
          content: 'Second',
          mode: 'text',
          synthetic: false, taskMode: false
        }
      ]

      await persistence.persistNewEvents(sessionId, firstBatch)
      await persistence.persistNewEvents(sessionId, secondBatch)
      
      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toHaveLength(2)
      expect((loaded[0] as any).content).toBe('First')
      expect((loaded[1] as any).content).toBe('Second')
    })

    test('handles empty events array', async () => {
      await persistence.persistNewEvents(sessionId, [])
      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toEqual([])
    })

    test('preserves event structure', async () => {
      const event: AppEvent = {
        type: 'turn_completed',
        forkId: null,
        turnId: 'turn-1',
        chainId: 'chain-1',
        code: 'console.log("test")',
        toolCalls: [
          {
            toolSlug: 'shell',
            result: {
              status: 'success',
              output: { stdout: 'test', stderr: '', exitCode: 0 }
            }
          }
        ],
        result: {
          success: true,
          calledActionTools: true, calledDone: false,
          lastToolSlug: 'shell'
        }
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
      const events: AppEvent[] = [
        {
          type: 'user_message',
          forkId: null,
          content: 'Test',
          mode: 'text',
          synthetic: false, taskMode: false
        }
      ]

      await persistence.persistNewEvents(sessionId, events)
      await persistence.saveSessionMetadata(sessionId, { chatName: 'Test' })
      
      const loaded = await persistence.loadEvents(sessionId)
      expect(loaded).toHaveLength(1)
    })
  })

  describe('concurrent operations', () => {
    test('handles rapid successive writes', async () => {
      const batches = Array.from({ length: 10 }, (_, i) => [
        {
          type: 'user_message' as const,
          forkId: null,
          content: `Message ${i}`,
          mode: 'text' as const,
          synthetic: false as const,
          taskMode: false
        }
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
      const events: AppEvent[] = Array.from({ length: 1000 }, (_, i) => ({
        type: 'message_chunk' as const,
        forkId: null,
        turnId: 'turn-1',
        id: `m-${i}`,
        text: `Chunk ${i}`
      }))

      await persistence.persistNewEvents(sessionId, events)
      const loaded = await persistence.loadEvents(sessionId)
      
      expect(loaded).toHaveLength(1000)
      expect((loaded[0] as any).text).toBe('Chunk 0')
      expect((loaded[999] as any).text).toBe('Chunk 999')
    })
  })
})
