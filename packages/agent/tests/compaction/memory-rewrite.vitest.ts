import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { baseContext, getMemory, mkCompactionCompleted, mkTurnCompleted, mkTurnStarted, mkUserMessage } from './helpers'

const sendAssistantText = (h: Effect.Effect.Success<typeof TestHarness>, turnId: string, text: string) =>
  Effect.gen(function* () {
    yield* h.send({ type: 'message_start', forkId: null, turnId, id: `${turnId}-msg`, destination: { kind: 'user' } })
    yield* h.send({ type: 'message_chunk', forkId: null, turnId, id: `${turnId}-msg`, text })
    yield* h.send({ type: 'message_end', forkId: null, turnId, id: `${turnId}-msg` })
  })

describe('compaction/memory-rewrite', () => {
  it.effect('compaction_completed rewrites head as [session_context, compacted, ...remaining]', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkUserMessage({ text: 'first message' }))
      yield* h.send(mkUserMessage({ text: 'second message' }))
      yield* h.send(mkCompactionCompleted({ summary: 'compacted summary', compactedMessageCount: 2, tokensSaved: 20 }))
      const memory = yield* getMemory(h)
      expect(memory.messages[0]?.type).toBe('session_context')
      expect(memory.messages[1]?.type).toBe('compacted')
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compacted message content preserves summary text exactly', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const summary = 'line 1\nline 2\nline 3'
      yield* h.send(mkCompactionCompleted({ summary, compactedMessageCount: 0 }))
      const memory = yield* getMemory(h)
      expect(memory.messages[1]?.type).toBe('compacted')
      if (memory.messages[1]?.type === 'compacted') {
        const text = memory.messages[1].content.map((p) => (p.type === 'text' ? p.text : '')).join('')
        expect(text).toBe(summary)
      }
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('remaining message ordering is preserved after rewrite', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      for (const id of [1, 2, 3]) {
        const turnId = `t-${id}`
        yield* h.send(mkTurnStarted({ turnId, chainId: 'chain-order' }))
        yield* sendAssistantText(h, turnId, `assistant-${id}`)
        yield* h.send(mkTurnCompleted({ turnId, chainId: 'chain-order' }))
      }
      const before = yield* getMemory(h)
      const compactedMessageCount = 2
      yield* h.send(mkCompactionCompleted({ summary: 's', compactedMessageCount }))
      const after = yield* getMemory(h)

      const beforeSuffix = before.messages.slice(1 + compactedMessageCount)
      const afterSuffix = after.messages.slice(2)
      expect(afterSuffix.length).toBe(beforeSuffix.length)
      for (let i = 0; i < beforeSuffix.length; i++) {
        expect(afterSuffix[i]).toBe(beforeSuffix[i])
      }
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('refreshedContext replaces existing session_context', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionCompleted({
        refreshedContext: baseContext({ cwd: '/tmp/new-cwd' }),
      }))
      const memory = yield* getMemory(h)
      expect(memory.messages[0]?.type).toBe('session_context')
      if (memory.messages[0]?.type === 'session_context') {
        const text = memory.messages[0].content.map((p) => (p.type === 'text' ? p.text : '')).join('')
        expect(text.includes('/tmp/new-cwd')).toBe(true)
      }
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('null refreshedContext preserves prior session_context', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const before = yield* getMemory(h)
      yield* h.send(mkCompactionCompleted({ refreshedContext: null }))
      const after = yield* getMemory(h)
      expect(after.messages[0]).toBe(before.messages[0])
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('currentChainId resets on compaction_completed', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 'turn-cid', chainId: 'chain-cid' }))
      yield* h.send(mkTurnCompleted({ turnId: 'turn-cid', chainId: 'chain-cid' }))
      yield* h.send(mkCompactionCompleted())
      const memory = yield* getMemory(h)
      expect(memory.currentChainId).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive())))
})