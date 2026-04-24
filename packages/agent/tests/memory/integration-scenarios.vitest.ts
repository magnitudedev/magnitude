import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getRootMemory, inboxMessages, snapshotMessageRefs, assertPrefixUnchanged, getRenderedUserText, sendUserMessage } from './helpers'

// Use getRenderedUserText from helpers instead of this function

describe('memory integration scenarios', () => {
  it.live('full conversation lifecycle', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'start task',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({ type: 'message_start', forkId: null, turnId: 't-1', id: 'm-t1', destination: { kind: 'user' } })
      yield* h.send({ type: 'message_chunk', forkId: null, turnId: 't-1', id: 'm-t1', text: 'first answer' })
      yield* h.send({ type: 'message_end', forkId: null, turnId: 't-1', id: 'm-t1' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        outcome: {
          _tag: 'Completed',
          completion: {
            yieldTarget: 'user',
            feedback: [{ _tag: 'InvalidMessageDestination', destination: 'unknown', message: 'follow-up reminder' }],
          },
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'continue',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = inboxMessages(memory)
      expect(inbox.length).toBeGreaterThan(1)

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('start task')
      expect(text).toContain('follow-up reminder')
      expect(text).toContain('continue')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' } })))
  )

  it.live('concurrent subagent activity during lead turn', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'agent_created',
        forkId: 'f-sub',
        parentForkId: null,
        agentId: 'builder-1',
        name: 'builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-1',
        message: 'go',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'message_start',
        forkId: 'f-sub',
        turnId: 'sub-turn',
        id: 'm1',
        destination: { kind: 'parent' },
      })
      yield* h.send({
        type: 'message_chunk',
        forkId: 'f-sub',
        turnId: 'sub-turn',
        id: 'm1',
        text: 'progress update',
      })
      yield* h.send({
        type: 'message_end',
        forkId: 'f-sub',
        turnId: 'sub-turn',
        id: 'm1',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'user ping',
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('user ping')
      expect(text).toContain('<agent ')
    }).pipe(Effect.provide(TestHarnessLive({ sessionContext: { timezone: 'UTC' } })))
  )

  it.live('mention resolution + tool execution + observation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'check file',
        attachments: [{ type: 'mention', path: 'src/x.ts', contentType: 'text' }],
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 't-1',
        parts: [{ type: 'text', text: 'observation' }],
      })
      yield* h.send({ type: 'message_start', forkId: null, turnId: 't-1', id: 'm-obs', destination: { kind: 'user' } })
      yield* h.send({ type: 'message_chunk', forkId: null, turnId: 't-1', id: 'm-obs', text: 'done' })
      yield* h.send({ type: 'message_end', forkId: null, turnId: 't-1', id: 'm-obs' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        outcome: { _tag: 'Completed', completion: { yieldTarget: 'user', feedback: [] } },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = inboxMessages(memory)
      expect(inbox.length).toBeGreaterThan(1)

      const firstUserInbox = inbox.find(m => m.type === 'inbox' && m.timeline.some(t => t.kind === 'user_message'))
      expect(firstUserInbox?.type).toBe('inbox')
      if (firstUserInbox?.type === 'inbox') {
        const userEntry = firstUserInbox.timeline.find(t => t.kind === 'user_message')
        expect(userEntry?.kind).toBe('user_message')
        if (userEntry?.kind === 'user_message') {
          expect(userEntry.attachments.length).toBe(1)
        }
      }

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('check file')
      expect(text).toContain('observation')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('error recovery remains append-only', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      const before = snapshotMessageRefs(yield* getRootMemory(h))

      yield* h.send({
        type: 'turn_outcome',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        outcome: { _tag: 'UnexpectedError', message: 'boom' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      const afterError = yield* getRootMemory(h)
      assertPrefixUnchanged(before, afterError)

      const beforeFollowup = snapshotMessageRefs(afterError)
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'retry now',
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const afterFollowup = yield* getRootMemory(h)
      assertPrefixUnchanged(beforeFollowup, afterFollowup)

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('boom')
      expect(text).toContain('retry now')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multi-turn queue coalescing with user message', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'while queued',
      })
      yield* h.send({ type: 'message_start', forkId: null, turnId: 't-1', id: 'm-done', destination: { kind: 'user' } })
      yield* h.send({ type: 'message_chunk', forkId: null, turnId: 't-1', id: 'm-done', text: 'assistant done' })
      yield* h.send({ type: 'message_end', forkId: null, turnId: 't-1', id: 'm-done' })
      yield* h.send({
        type: 'turn_outcome',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        outcome: { _tag: 'Completed', completion: { yieldTarget: 'user', feedback: [] } },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const text = yield* getRenderedUserText(h)
      expect(text).toContain('while queued')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
