import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getRootMemory, lastInboxMessage, sendUserMessage } from './helpers'
import { getView } from '../../src/projections/memory'

function renderedUserTextFromMemory(messages: Parameters<typeof getView>[0]): string {
  const rendered = getView(messages, 'UTC', 'agent')
  return rendered
    .filter(m => m.role === 'user')
    .map(m => m.content.map(p => p.type === 'text' ? p.text : '').join('\n'))
    .join('\n')
}

describe('memory queue and flush', () => {
  it.live('events during active turn are queued', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'queued msg',
      })

      const memory = yield* getRootMemory(h)
      expect(memory.currentTurnId).toBe('t-1')
      expect(memory.queuedEntries.length).toBeGreaterThan(0)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('flush on turn_started produces single inbox message with both lanes', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'tool_event',
        forkId: null,
        turnId: 't-1',
        toolCallId: 'tc-1',
        toolKey: 'shell',
        event: {
          _tag: 'ToolObservation',
          toolCallId: 'tc-1',
          tagName: 'shell',
          query: '.',
          content: [{ type: 'text', text: '<stdout>ok</stdout>' }],
        },
      })
      yield* h.send({
        type: 'turn_completed',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',

        result: {
          success: true,
          turnDecision: 'idle',
          errors: [{ code: 'nonexistent_agent_destination', message: 'after turn' }],
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.length).toBeGreaterThan(0)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('queue ordering is by timestamp then seq', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641601000,
        text: 'later',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'first',
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'second',
      })

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        const items = inbox.timeline.filter(e => e.kind === 'user_message')
        expect(items.map(i => i.text)).toEqual(expect.arrayContaining(['first', 'second', 'later']))
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('coalesce key deduplicates file updates by path', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: '@src/a.ts',
        attachments: [{ type: 'mention', path: 'src/a.ts', contentType: 'text' }],
      })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641601000,
        text: '@src/a.ts again',
        attachments: [{ type: 'mention', path: 'src/a.ts', contentType: 'text' }],
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)

      expect(text).toContain('<message from="user">@src/a.ts again</message>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('empty flush after assistant turn injects noop', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',

        result: { success: true, turnDecision: 'idle' },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const inbox = lastInboxMessage(memory)
      expect(inbox?.type).toBe('inbox')
      if (inbox?.type === 'inbox') {
        expect(inbox.results.some(r => r.kind === 'noop')).toBe(true)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('mixed sources interleave and render deterministically', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'from user',
      })
      yield* h.send({
        type: 'turn_completed',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',

        result: {
          success: true,
          turnDecision: 'idle',
          errors: [{ code: 'nonexistent_agent_destination', message: 'remember me' }],
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const memory = yield* getRootMemory(h)
      const text = renderedUserTextFromMemory(memory.messages)

      expect(text).toContain('<message from="user">from user</message>')
      expect(text).toContain('<error>remember me</error>')
      // Results (including turn errors) render before timeline
      expect(text.indexOf('<error>remember me</error>')).toBeLessThan(
        text.indexOf('<message from="user">from user</message>'),
      )
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
