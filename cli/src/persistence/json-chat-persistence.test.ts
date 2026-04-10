import { describe, expect, test } from 'vitest'
import { Effect } from 'effect'
import { JsonChatPersistence } from './json-chat-persistence'
import { CLI_VERSION } from '../version'
import type { StoredSessionMeta, StorageClient } from '@magnitudedev/storage'

function createStorage(seed?: StoredSessionMeta): StorageClient {
  let metadata = seed ?? null
  const events: unknown[] = []

  return {
    sessions: {
      createId: () => 'session-1',
      list: async () => [],
      findLatest: async () => null,
      readMeta: async () => metadata,
      writeMeta: async (_sessionId: string, meta: StoredSessionMeta) => {
        metadata = meta
      },
      updateMeta: async (_sessionId: string, updater: (current: StoredSessionMeta | null) => StoredSessionMeta) => {
        metadata = updater(metadata)
        return metadata
      },
      readEvents: async () => events,
      appendEvents: async (_sessionId: string, nextEvents: unknown[]) => {
        events.push(...nextEvents)
      },
      getEventsPath: () => '/tmp/session-1/events.jsonl',
      createWorkspace: async () => '/tmp/session-1',
      getWorkspacePath: () => '/tmp/session-1',
    },
  } as any
}

describe('JsonChatPersistence', () => {
  test('persistNewEvents writes initialVersion and lastActiveVersion on first metadata write', async () => {
    const storage = createStorage()
    const persistence = new JsonChatPersistence({
      storage,
      workingDirectory: '/repo',
      sessionId: 'session-1',
    })

    await Effect.runPromise(
      persistence.persistNewEvents([
        { type: 'user_message', content: [{ type: 'text', text: 'hello' }] } as any,
      ])
    )

    const metadata = await storage.sessions.readMeta('session-1')

    expect(metadata?.initialVersion).toBe(CLI_VERSION)
    expect(metadata?.lastActiveVersion).toBe(CLI_VERSION)
  })

  test('saveSessionMetadata preserves initialVersion and refreshes lastActiveVersion', async () => {
    const storage = createStorage({
      sessionId: 'session-1',
      created: '2026-01-01T00:00:00.000Z',
      updated: '2026-01-01T00:00:00.000Z',
      chatName: 'Old',
      workingDirectory: '/repo',
      initialVersion: '0.0.0',
      lastActiveVersion: '0.0.0',
      gitBranch: null,
      firstUserMessage: null,
      lastMessage: null,
      messageCount: 0,
    })
    const persistence = new JsonChatPersistence({
      storage,
      workingDirectory: '/repo',
      sessionId: 'session-1',
    })

    await Effect.runPromise(persistence.saveSessionMetadata({ chatName: 'New' }))

    const metadata = await storage.sessions.readMeta('session-1')
    expect(metadata?.initialVersion).toBe('0.0.0')
    expect(metadata?.lastActiveVersion).toBe(CLI_VERSION)
    expect(metadata?.chatName).toBe('New')
  })

  test('getSessionMetadata exposes version fields', async () => {
    const storage = createStorage({
      sessionId: 'session-1',
      created: '2026-01-01T00:00:00.000Z',
      updated: '2026-01-01T00:00:00.000Z',
      chatName: 'Chat',
      workingDirectory: '/repo',
      initialVersion: '0.0.0',
      lastActiveVersion: '0.0.1',
      gitBranch: null,
      firstUserMessage: null,
      lastMessage: null,
      messageCount: 0,
    })
    const persistence = new JsonChatPersistence({
      storage,
      workingDirectory: '/repo',
      sessionId: 'session-1',
    })

    const metadata = await Effect.runPromise(persistence.getSessionMetadata())

    expect(metadata.initialVersion).toBe('0.0.0')
    expect(metadata.lastActiveVersion).toBe('0.0.1')
  })
})
