import { FetchHttpClient } from '@effect/platform'
import { Effect, Layer, Stream } from 'effect'
import { describe, expect, it } from 'vitest'
import { EventEngine } from '@magnitudedev/event-core'
import {
  ModelStreamTerminal,
  StreamClientCorrectnessViolation,
  type BaseCallOptions,
  type ModelStreamResult,
  type ResponseStreamEvent,
} from '@magnitudedev/ai'

import { DEFAULT_CHAT_NAME } from '../src/constants'
import type { AppEvent } from '../src/events'
import type { AgentBoundModel } from '../src/model/agent-model'
import { AgentModelResolver } from '../src/model/model-resolver'
import {
  ChatPersistence,
  type ChatPersistenceService,
  type SessionMetadata,
} from '../src/persistence/chat-persistence-service'
import { ChatTitleProjection } from '../src/projections/chat-title'
import { UserMessageResolutionProjection } from '../src/projections/user-message-resolution'
import { fallbackChatTitle } from '../src/util/title-prompts'
import {
  CHAT_TITLE_MAX_TOKENS,
  ChatTitleServiceLive,
} from '../src/workers/chat-title-service'
import { ChatTitleWorker } from '../src/workers/chat-title-worker'

const TestAgent = EventEngine.make<AppEvent>()({
  name: 'ChatTitleTestAgent',
  schemaVersion: 'test',
  projections: [UserMessageResolutionProjection, ChatTitleProjection],
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

function userMessage(messageId: string, text: string): AppEvent {
  return {
    type: 'user_message',
    forkId: null,
    messageId,
    timestamp: Date.now(),
    content: [{ _tag: 'TextPart', text }],
    attachments: [],
    mode: 'text',
    synthetic: false,
    taskMode: false,
  }
}

function userMessageReady(messageId: string): AppEvent {
  return {
    type: 'user_message_ready',
    forkId: null,
    messageId,
    mentionResolutions: [],
  }
}

function makePersistence() {
  const now = new Date().toISOString()
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
      metadata = { ...metadata, ...update, updated: new Date().toISOString() }
    }),
  }

  return {
    layer: Layer.succeed(ChatPersistence, service),
    metadata: () => metadata,
  }
}

function makeTitleModel(title: string | null, options?: { failAfterOutput?: boolean }) {
  let primaryResolutions = 0
  let secondaryResolutions = 0
  let streamCalls = 0
  let maxTokens: number | undefined

  const events: ResponseStreamEvent[] = title === null
    ? []
    : [{ _tag: 'message_delta', text: title }]
  if (options?.failAfterOutput) {
    events.push({
      _tag: 'stream_end',
      terminal: ModelStreamTerminal.StreamFailed({
        cause: new StreamClientCorrectnessViolation({
          call: { provider: 'test', model: 'local-primary', method: 'POST', url: 'http://test' },
          response: { status: 200, headers: [], requestId: null },
          component: 'model_event_reducer',
          message: 'stream ended before the grammar root accepted',
          evidence: {
            _tag: 'InvariantViolated',
            invariant: 'grammar root must be accepted before the stream ends',
          },
          progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 1 },
        }),
        usage: { _tag: 'UsageNotReported', reason: 'stream_failed_before_usage' },
      }),
    })
  }
  const result: ModelStreamResult = {
    events: Stream.fromIterable(events),
    parsers: new Map(),
    logprobs: [],
    requestId: null,
  }
  const model: AgentBoundModel = {
    model: {
      stream: (_prompt, _tools, options?: BaseCallOptions) => Effect.sync(() => {
        streamCalls += 1
        maxTokens = options?.maxTokens
        return result
      }),
    },
    modelSource: { slotId: 'primary' },
    modelId: 'local-primary',
    providerId: 'llamacpp',
    profile: {
      contextWindow: 128_000,
      maxOutputTokens: 4096,
    },
  }

  const layer = Layer.succeed(AgentModelResolver, {
    resolvePrimary: () => Effect.sync(() => {
      primaryResolutions += 1
      return model
    }),
    resolveSecondary: () => Effect.sync(() => {
      secondaryResolutions += 1
      return model
    }),
  })

  return {
    layer,
    counts: () => ({ primaryResolutions, secondaryResolutions, streamCalls, maxTokens }),
  }
}

function requirements(
  persistenceLayer: Layer.Layer<ChatPersistence>,
  resolverLayer: Layer.Layer<AgentModelResolver>,
) {
  const titleServiceLayer = ChatTitleServiceLive.pipe(
    Layer.provide(Layer.merge(resolverLayer, FetchHttpClient.layer)),
  )
  return Layer.merge(persistenceLayer, titleServiceLayer) as Parameters<typeof TestAgent.createClient>[0]
}

describe('chat title generation', () => {
  it('creates concise fallback titles', () => {
    expect(fallbackChatTitle('  fix   the broken login flow  ')).toBe('Fix the broken login flow')
    const title = fallbackChatTitle('investigate why every single message starts another secondary model title request')
    expect(title).toBe('Investigate why every single message starts…')
    expect(title!.length).toBeLessThanOrEqual(50)
    expect(fallbackChatTitle('   ')).toBeNull()
  })

  it('uses the primary model once and keeps the fallback when generation fails', async () => {
    const persistence = makePersistence()
    const titleModel = makeTitleModel(null)
    const client = await TestAgent.createClient(requirements(persistence.layer, titleModel.layer))
    const events: AppEvent[] = []
    const unsubscribe = client.onEvent((event) => events.push(event))

    try {
      await client.send(userMessage('message-1', 'debug duplicate model requests'))
      await client.send(userMessageReady('message-1'))
      await waitFor(() => titleModel.counts().streamCalls === 1)

      expect(persistence.metadata().chatName).toBe('Debug duplicate model requests')
      expect(events.some((event) =>
        event.type === 'chat_title_generated'
        && event.title === 'Debug duplicate model requests')).toBe(true)
      expect(titleModel.counts()).toEqual({
        primaryResolutions: 1,
        secondaryResolutions: 0,
        streamCalls: 1,
        maxTokens: CHAT_TITLE_MAX_TOKENS,
      })

      await client.send(userMessage('message-2', 'this must not generate another title'))
      await client.send(userMessageReady('message-2'))
      await sleep(50)

      expect(titleModel.counts().streamCalls).toBe(1)
    } finally {
      unsubscribe()
      await client.dispose()
    }
  })

  it('replaces the fallback when the primary model returns a title', async () => {
    const persistence = makePersistence()
    const titleModel = makeTitleModel('Debug duplicate title calls')
    const client = await TestAgent.createClient(requirements(persistence.layer, titleModel.layer))

    try {
      await client.send(userMessage('message-1', 'please figure out these repeated background requests'))
      await client.send(userMessageReady('message-1'))
      await waitFor(() => persistence.metadata().chatName === 'Debug duplicate title calls')

      expect(titleModel.counts().streamCalls).toBe(1)
      expect(titleModel.counts().secondaryResolutions).toBe(0)
    } finally {
      await client.dispose()
    }
  })

  it('uses generated title text even when the stream truncates afterward', async () => {
    const persistence = makePersistence()
    const titleModel = makeTitleModel('Investigate repeated background calls', { failAfterOutput: true })
    const client = await TestAgent.createClient(requirements(persistence.layer, titleModel.layer))

    try {
      await client.send(userMessage('message-1', 'why does this call happen every time'))
      await client.send(userMessageReady('message-1'))
      await waitFor(() => persistence.metadata().chatName === 'Investigate repeated background calls')

      expect(titleModel.counts().streamCalls).toBe(1)
    } finally {
      await client.dispose()
    }
  })
})
