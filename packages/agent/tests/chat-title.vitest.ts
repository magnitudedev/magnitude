import { Effect, Layer } from 'effect'
import { describe, expect, it } from 'vitest'
import { EventEngine } from '@magnitudedev/event-core'

import { DEFAULT_CHAT_NAME } from '../src/constants'
import type { AppEvent } from '../src/events'
import {
  ChatPersistence,
  type ChatPersistenceService,
  type SessionMetadata,
} from '../src/persistence/chat-persistence-service'
import { ChatTitleProjection } from '../src/projections/chat-title'
import {
  CHAT_TITLE_MAX_CHARACTERS,
  deriveChatTitle,
} from '../src/util/chat-title'
import { ChatTitleWorker } from '../src/workers/chat-title-worker'

const TestAgent = EventEngine.make<AppEvent>()({
  name: 'ChatTitleTestAgent',
  schemaVersion: 'test',
  projections: [ChatTitleProjection],
  workers: [ChatTitleWorker],
})

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function waitFor(assertion: () => boolean, timeoutMs = 1000) {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    if (assertion()) return
    await sleep(10)
  }
  throw new Error('Timed out waiting for condition')
}

function userMessage(
  messageId: string,
  text: string,
  options?: { readonly forkId?: string | null; readonly synthetic?: boolean },
): AppEvent {
  return {
    type: 'user_message',
    forkId: options?.forkId ?? null,
    messageId,
    timestamp: Date.now(),
    text,
    mentions: [],
    attachments: [],
    mode: 'text',
    synthetic: options?.synthetic ?? false,
    taskMode: false,
  }
}

function makePersistence() {
  const now = new Date().toISOString()
  let saveCount = 0
  let metadata: SessionMetadata = {
    sessionId: 'chat-title-test',
    chatName: DEFAULT_CHAT_NAME,
    workingDirectory: process.cwd(),
    gitBranch: null,
    created: now,
    updated: now,
    initialVersion: 'test',
    lastActiveVersion: 'test',
  }

  const service: ChatPersistenceService = {
    loadEvents: () => Effect.succeed([]),
    loadEventsAfterCursor: () => Effect.succeed([]),
    persistNewEvents: () => Effect.succeed(null),
    loadProjectionSnapshot: () => Effect.succeed(null),
    saveProjectionSnapshot: () => Effect.void,
    getSessionMetadata: () => Effect.sync(() => metadata),
    saveSessionMetadata: (update) => Effect.sync(() => {
      saveCount += 1
      metadata = { ...metadata, ...update, updated: new Date().toISOString() }
    }),
  }

  return {
    layer: Layer.succeed(ChatPersistence, service),
    metadata: () => metadata,
    saveCount: () => saveCount,
  }
}

describe('chat titles', () => {
  it('normalizes whitespace and keeps the first 50 characters', () => {
    expect(deriveChatTitle('  fix\n\tthe   broken login flow  ')).toBe('fix the broken login flow')

    const longMessage = '0123456789'.repeat(6)
    expect(deriveChatTitle(longMessage)).toBe('0123456789'.repeat(5))
    expect(Array.from(deriveChatTitle(longMessage)!).length).toBe(CHAT_TITLE_MAX_CHARACTERS)
    expect(deriveChatTitle('   \n\t  ')).toBeNull()
  })

  it('counts Unicode code points without splitting an emoji', () => {
    const message = `${'a'.repeat(CHAT_TITLE_MAX_CHARACTERS - 1)}🙂suffix`
    expect(deriveChatTitle(message)).toBe(`${'a'.repeat(CHAT_TITLE_MAX_CHARACTERS - 1)}🙂`)
  })

  it('persists a title from the first real root user message only', async () => {
    const persistence = makePersistence()
    const client = await TestAgent.createClient(persistence.layer)

    try {
      await client.send(userMessage('forked', 'fork title', { forkId: 'fork-1' }))
      await client.send(userMessage('synthetic', 'synthetic title', { synthetic: true }))
      await client.send(userMessage('first', '  first\n real   title  '))
      await waitFor(() => persistence.metadata().chatName === 'first real title')

      await client.send(userMessage('second', 'second title must be ignored'))
      await sleep(50)

      expect(persistence.metadata().chatName).toBe('first real title')
      expect(persistence.saveCount()).toBe(1)
    } finally {
      await client.dispose()
    }
  })

  it('resolves a blank first message without using a later message', async () => {
    const persistence = makePersistence()
    const client = await TestAgent.createClient(persistence.layer)

    try {
      await client.send(userMessage('blank', '  \n '))
      await client.send(userMessage('second', 'must not become the title'))
      await sleep(50)

      expect(persistence.metadata().chatName).toBe(DEFAULT_CHAT_NAME)
      expect(persistence.saveCount()).toBe(0)
    } finally {
      await client.dispose()
    }
  })
})
