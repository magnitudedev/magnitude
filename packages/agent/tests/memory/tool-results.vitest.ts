import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getRootMemory, lastInboxMessage } from './helpers'
import { getView } from '../../src/projections/memory'

function renderedUserTextFromMemory(messages: Parameters<typeof getView>[0]): string {
  const rendered = getView(messages, 'UTC', 'agent')
  return rendered
    .filter(m => m.role === 'user')
    .map(m => m.content.map(p => p.type === 'text' ? p.text : '').join('\n'))
    .join('\n')
}

describe('memory tool results', () => {
  it.live('single tool call renders result', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [{ toolKey: 'shell', group: 'shell', toolName: 'shell', result: { status: 'success', output: { ok: true } } }],
        observedResults: [{ toolCallId: 'tc-1', tagName: 'shell', query: '.', content: [{ type: 'text', text: '<stdout>hi</stdout>' }] }],
        result: { success: true, turnDecision: 'yield' },
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
      expect(text).toContain('<shell')
      expect(text).toContain('<stdout>hi</stdout>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multiple tool calls render all results', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [
          { toolKey: 'shell', group: 'shell', toolName: 'shell', result: { status: 'success', output: {} } },
          { toolKey: 'shell', group: 'shell', toolName: 'shell', result: { status: 'success', output: {} } },
        ],
        observedResults: [
          { toolCallId: 'tc-a', tagName: 'shell', query: '.', content: [{ type: 'text', text: '<stdout>a</stdout>' }] },
          { toolCallId: 'tc-b', tagName: 'shell', query: '.', content: [{ type: 'text', text: '<stdout>b</stdout>' }] },
        ],
        result: { success: true, turnDecision: 'yield' },
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
      expect(text).toContain('<stdout>a</stdout>')
      expect(text).toContain('<stdout>b</stdout>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('tool error is rendered', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [],
        observedResults: [],
        result: { success: false, error: 'boom', cancelled: false },
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
        const tr = inbox.results.find(r => r.kind === 'tool_results')
        const err = inbox.results.find(r => r.kind === 'error')
        expect(tr).toBeUndefined()
        expect(err?.kind).toBe('error')
        if (err?.kind === 'error') {
          expect(err.message).toContain('boom')
        }
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('interrupted turn renders interrupted result', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [],
        observedResults: [],
        result: { success: false, error: 'cancelled', cancelled: true },
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
        expect(inbox.results.some(r => r.kind === 'interrupted')).toBe(true)
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('noop result emitted after assistant turn and empty flush', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'assistant text' }],
        toolCalls: [],
        observedResults: [],
        result: { success: true, turnDecision: 'yield' },
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

  it.live('large output shows truncation guidance', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const large = 'x'.repeat(30000)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'turn_completed',
        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        responseParts: [{ type: 'text', content: 'done' }],
        toolCalls: [{ toolKey: 'shell', group: 'shell', toolName: 'shell', result: { status: 'success', output: {} } }],
        observedResults: [{ toolCallId: 'tc-large', tagName: 'shell', query: '.', content: [{ type: 'text', text: large }] }],
        result: { success: true, turnDecision: 'yield' },
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
      expect(text).toContain('<shell observe=".">')
      expect(text).toContain('xxxxxxxx')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
