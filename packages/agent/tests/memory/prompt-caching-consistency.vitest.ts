import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { assertPrefixUnchanged, getRenderedUserText, getRootMemory, sendUserMessage, snapshotMessageRefs } from './helpers'

// Use getRenderedUserText from helpers instead of this function

describe('prompt caching consistency', () => {
  it.live('user messages append without rewriting prior rendered prefix', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'one',
      })

      const beforeRendered = yield* getRenderedUserText(h)
      const beforeMemory = yield* getRootMemory(h)
      const before = snapshotMessageRefs(beforeMemory)

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641660000,
        text: 'two',
      })

      const afterRendered = yield* getRenderedUserText(h)
      const afterMemory = yield* getRootMemory(h)

      expect(afterRendered).toContain(beforeRendered)
      assertPrefixUnchanged(before, afterMemory)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('observations append new inbox and preserve prefix', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'base',
      })

      const beforeRendered = yield* getRenderedUserText(h)
      const beforeMemory = yield* getRootMemory(h)
      const before = snapshotMessageRefs(beforeMemory)

      yield* h.send({
        type: 'observations_captured',
        forkId: null,
        turnId: 't-1',
        parts: [{ type: 'text', text: 'observation note' }],
      })

      const afterRendered = yield* getRenderedUserText(h)
      const afterMemory = yield* getRootMemory(h)

      expect(afterRendered).toContain(beforeRendered)
      expect(afterRendered).toContain('observation note')
      assertPrefixUnchanged(before, afterMemory)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('turn unexpected error appends new inbox and preserves prefix', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'base',
      })

      const beforeRendered = yield* getRenderedUserText(h)
      const beforeMemory = yield* getRootMemory(h)
      const before = snapshotMessageRefs(beforeMemory)

      yield* h.send({
        type: 'turn_unexpected_error',
        forkId: null,
        turnId: 't-1',
        message: 'fatal failure',
      })

      const afterRendered = yield* getRenderedUserText(h)
      const afterMemory = yield* getRootMemory(h)

      expect(afterRendered).toContain(beforeRendered)
      expect(afterRendered).toContain('fatal failure')
      assertPrefixUnchanged(before, afterMemory)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('queue coalescing rewrites queue only and flushes latest update', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'base',
      })

      const beforeMemory = yield* getRootMemory(h)
      const before = snapshotMessageRefs(beforeMemory)

      yield* h.send({ type: 'tool_event', forkId: null, turnId: 't-1', toolCallId: 'x', toolKey: 'fileRead', event: { _tag: 'ToolInputStarted', toolCallId: 'x', tagName: 'read', toolName: 'fileRead', group: 'fs',  } })
      yield* h.send({ type: 'tool_event', forkId: null, turnId: 't-1', toolCallId: 'x', toolKey: 'fileRead', event: { _tag: 'ToolInputStarted', toolCallId: 'x', tagName: 'read', toolName: 'fileRead', group: 'fs',  } })

      const midMemory = yield* getRootMemory(h)
      assertPrefixUnchanged(before, midMemory)

      yield* h.send({
        type: 'turn_completed',

        forkId: null,
        turnId: 't-1',
        chainId: 'c-1',
        strategyId: 'xml-act',
        result: {
          success: true,
          turnDecision: 'idle',
          errors: [{ code: 'nonexistent_agent_destination', message: 'latest only' }],
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const afterRendered = yield* getRenderedUserText(h)
      expect(afterRendered).toContain('latest only')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('flush appends one inbox without rewriting prior messages', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-1', chainId: 'c-1' })
      yield* h.send({
        type: 'skill_started',
        forkId: null,
        source: 'user',
        skill: { name: 's', description: '', preamble: '', phases: [{ name: 'p', prompt: 'x' }] },
      })

      const beforeRendered = yield* getRenderedUserText(h)
      const beforeMemory = yield* getRootMemory(h)
      const before = snapshotMessageRefs(beforeMemory)

      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-2', chainId: 'c-1' })

      const afterRendered = yield* getRenderedUserText(h)
      const afterMemory = yield* getRootMemory(h)
      expect(afterRendered).toContain(beforeRendered)
      assertPrefixUnchanged(before, afterMemory)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('compaction is explicit exception and rewrites history', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* sendUserMessage(h, {
        forkId: null,
        timestamp: 1711641600000,
        text: 'before compaction',
      })
      yield* h.send({ type: 'turn_started', forkId: null, turnId: 't-compaction-flush', chainId: 'c-compaction-flush' })

      const beforeRendered = yield* getRenderedUserText(h)

      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'summary block',
        compactedMessageCount: 1,
        tokensSaved: 100,
        preservedVariables: [],
        refreshedContext: null,
      })

      const afterRendered = yield* getRenderedUserText(h)
      expect(beforeRendered).toContain('before compaction')
      expect(afterRendered).toContain('summary block')
      expect(afterRendered).toContain('<session_context>')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
