/**
 * Persistence Tests
 *
 * Tests for ChatPersistenceService implementations.
 */

import { describe, test, expect, beforeEach } from 'bun:test'
import { createInMemoryPersistence } from './in-memory-persistence'
import { createJsonFilePersistence } from './json-file-persistence'
import { createId } from '@paralleldrive/cuid2'
import { rm } from 'node:fs/promises'
import { join } from 'node:path'
import { homedir } from 'node:os'
import type { AppEvent } from '@magnitudedev/agent'

// Test events
const createTestEvent = (type: string, forkId: string | null = null): AppEvent => ({
  type: type as any,
  forkId,
  timestamp: Date.now()
} as any)

describe('InMemoryPersistence', () => {
  test('should initialize with empty events', async () => {
    const persistence = createInMemoryPersistence(
      createId(),
      '/test/dir',
      'main'
    )

    const events = await persistence.loadEvents()
    expect(events).toEqual([])
    expect(await persistence.exists()).toBe(false)
  })

  test('should persist and load events', async () => {
    const persistence = createInMemoryPersistence(
      createId(),
      '/test/dir',
      'main'
    )

    const testEvents = [
      createTestEvent('session_initialized'),
      createTestEvent('user_message'),
      createTestEvent('turn_started')
    ]

    await persistence.persistNewEvents(testEvents)
    
    const loaded = await persistence.loadEvents()
    expect(loaded).toHaveLength(3)
    expect(loaded[0].type).toBe('session_initialized')
  })

  test('should update metadata', async () => {
    const persistence = createInMemoryPersistence(
      createId(),
      '/test/dir',
      'main'
    )

    const initialMeta = await persistence.getSessionMetadata()
    expect(initialMeta.chatName).toBeNull()

    await persistence.saveSessionMetadata({ chatName: 'Test Chat' })

    const updatedMeta = await persistence.getSessionMetadata()
    expect(updatedMeta.chatName).toBe('Test Chat')
    // Updated timestamp should be >= initial (may be same if too fast)
    expect(new Date(updatedMeta.updated).getTime()).toBeGreaterThanOrEqual(
      new Date(initialMeta.updated).getTime()
    )
  })

  test('should append events across multiple calls', async () => {
    const persistence = createInMemoryPersistence(
      createId(),
      '/test/dir',
      'main'
    )

    await persistence.persistNewEvents([createTestEvent('event1')])
    await persistence.persistNewEvents([createTestEvent('event2')])
    await persistence.persistNewEvents([createTestEvent('event3')])

    const loaded = await persistence.loadEvents()
    expect(loaded).toHaveLength(3)
  })
})

describe('JsonFilePersistence', () => {
  const sessionDir = join(homedir(), '.magnitude', 'sessions')
  let testSessionId: string

  beforeEach(() => {
    testSessionId = `test-${createId()}`
  })

  const cleanup = async () => {
    try {
      await rm(join(sessionDir, `${testSessionId}.json`), { force: true })
      await rm(join(sessionDir, `${testSessionId}.json.tmp`), { force: true })
    } catch {
      // Ignore errors
    }
  }

  test('should persist events to file', async () => {
    await cleanup()

    const persistence = createJsonFilePersistence(
      testSessionId,
      '/test/dir',
      'main'
    )

    const testEvents = [
      createTestEvent('session_initialized'),
      createTestEvent('user_message')
    ]

    await persistence.persistNewEvents(testEvents)
    expect(await persistence.exists()).toBe(true)

    const loaded = await persistence.loadEvents()
    expect(loaded).toHaveLength(2)

    await cleanup()
  })

  test('should handle multiple persist calls', async () => {
    await cleanup()

    const persistence = createJsonFilePersistence(
      testSessionId,
      '/test/dir',
      'main'
    )

    await persistence.persistNewEvents([createTestEvent('event1')])
    await persistence.persistNewEvents([createTestEvent('event2')])
    await persistence.persistNewEvents([createTestEvent('event3')])

    const loaded = await persistence.loadEvents()
    expect(loaded).toHaveLength(3)

    await cleanup()
  })

  test('should update metadata in file', async () => {
    await cleanup()

    const persistence = createJsonFilePersistence(
      testSessionId,
      '/test/dir',
      'main'
    )

    await persistence.persistNewEvents([createTestEvent('event1')])
    await persistence.saveSessionMetadata({ chatName: 'Updated Name' })

    const metadata = await persistence.getSessionMetadata()
    expect(metadata.chatName).toBe('Updated Name')

    await cleanup()
  })

  test('should return empty array for non-existent session', async () => {
    await cleanup()

    const persistence = createJsonFilePersistence(
      testSessionId,
      '/test/dir',
      'main'
    )

    const events = await persistence.loadEvents()
    expect(events).toEqual([])
    expect(await persistence.exists()).toBe(false)

    await cleanup()
  })

  test('should handle sequential writes correctly', async () => {
    await cleanup()

    const persistence = createJsonFilePersistence(
      testSessionId,
      '/test/dir',
      'main'
    )

    // Sequential writes (not concurrent - that would require locking)
    await persistence.persistNewEvents([createTestEvent('event1')])
    await persistence.persistNewEvents([createTestEvent('event2')])
    await persistence.persistNewEvents([createTestEvent('event3')])

    const loaded = await persistence.loadEvents()
    expect(loaded).toHaveLength(3)
    expect(loaded[0].type).toBe('event1')
    expect(loaded[1].type).toBe('event2')
    expect(loaded[2].type).toBe('event3')

    await cleanup()
  })
})
