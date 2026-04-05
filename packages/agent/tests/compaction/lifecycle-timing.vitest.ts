import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getCompaction, getTurn, mkCompactionReady, mkCompactionStarted, mkContextLimitHit, mkTurnCompleted, mkTurnStarted } from './helpers'

describe('compaction/lifecycle-timing', () => {
  it.effect('turn completes during summarization window does not prevent finalization', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkTurnStarted({ turnId: 't1', chainId: 'c1' }))
      yield* h.send(mkTurnCompleted({ turnId: 't1', chainId: 'c1' }))
      yield* h.send(mkCompactionReady())
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'summary',
        compactedMessageCount: 1,
        tokensSaved: 10,
        preservedVariables: [],
        refreshedContext: null,
      })
      const state = yield* getCompaction(h)
      expect(state._tag === 'pendingFinalization').toBe(false)
      expect(state._tag !== 'idle').toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_ready while turn in-flight defers finalize until turn_completed', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({
        type: 'user_message',
        messageId: 'm-defers',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'X'.repeat(60_000) }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })
      yield* h.send(mkTurnStarted({ turnId: 'A', chainId: 'C' }))
      yield* h.send(mkContextLimitHit())
      yield* h.wait.event('compaction_ready', (e) => e.forkId === null)
      const compactionWhileTurnInFlight = yield* getCompaction(h)
      expect(compactionWhileTurnInFlight._tag).toBe('idle')

      yield* h.send(mkTurnCompleted({ turnId: 'A', chainId: 'C' }))

      yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      const after = yield* getCompaction(h)
      expect(after._tag !== 'idle').toBe(false)
      expect(after.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true }, model: { completeResponse: 'worker summary' } }))))

  it.effect('compaction_ready while idle finalizes immediately with worker', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({
        type: 'user_message',
        messageId: 'm1',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'X'.repeat(60_000) }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })
      yield* h.send(mkContextLimitHit())
      const ready = yield* h.wait.event('compaction_ready', (e) => e.forkId === null)
      const completed = yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      expect(completed.summary).toBe(ready.summary)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true }, model: { completeResponse: 'worker summary' } }))))

  it.effect('finalize timing parity immediate vs deferred terminal state', () =>
    Effect.gen(function* () {
      const immediate = yield* Effect.gen(function* () {
        const hImmediate = yield* TestHarness
        yield* hImmediate.send({
          type: 'user_message',
          messageId: 'm-parity-immediate',
          forkId: null,
          timestamp: Date.now(),
          content: [{ type: 'text', text: 'X'.repeat(60_000) }],
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
        yield* hImmediate.send(mkTurnStarted({ turnId: 'parity-immediate-turn', chainId: 'parity-immediate-chain' }))
        yield* hImmediate.send(mkTurnCompleted({ turnId: 'parity-immediate-turn', chainId: 'parity-immediate-chain' }))
        yield* hImmediate.send(mkContextLimitHit())
        yield* hImmediate.wait.event('compaction_ready', (e) => e.forkId === null)
        yield* hImmediate.wait.event('compaction_completed', (e) => e.forkId === null)
        return yield* getCompaction(hImmediate)
      }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true, turnController: false }, model: { completeResponse: 'worker summary' } })))

      const deferred = yield* Effect.gen(function* () {
        const hDeferred = yield* TestHarness
        yield* hDeferred.send({
          type: 'user_message',
          messageId: 'm-parity-deferred',
          forkId: null,
          timestamp: Date.now(),
          content: [{ type: 'text', text: 'X'.repeat(60_000) }],
          attachments: [],
          mode: 'text',
          synthetic: false,
          taskMode: false,
        })
        yield* hDeferred.send(mkTurnStarted({ turnId: 'parity-turn', chainId: 'parity-chain' }))
        yield* hDeferred.send(mkContextLimitHit())
        yield* hDeferred.wait.event('compaction_ready', (e) => e.forkId === null)
        yield* hDeferred.send(mkTurnCompleted({ turnId: 'parity-turn', chainId: 'parity-chain' }))
        yield* hDeferred.wait.event('compaction_completed', (e) => e.forkId === null)
        return yield* getCompaction(hDeferred)
      }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true, turnController: false }, model: { completeResponse: 'worker summary' } })))

      expect(immediate.contextLimitBlocked).toBe(false)
      expect(immediate._tag !== 'idle').toBe(false)
      expect(deferred.contextLimitBlocked).toBe(immediate.contextLimitBlocked)
      expect(deferred._tag !== 'idle').toBe(immediate._tag !== 'idle')
    }))

  it.effect('multiple context_limit_hit are idempotent with worker', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({
        type: 'user_message',
        messageId: 'm2',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'X'.repeat(60_000) }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkContextLimitHit())
      yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      const events = h.events().filter((e) => e.forkId === null)
      const firstStarted = events.findIndex((e) => e.type === 'compaction_started')
      const firstTerminal = events.findIndex((e) => e.type === 'compaction_completed' || e.type === 'compaction_failed')
      expect(firstStarted).toBeGreaterThanOrEqual(0)
      expect(firstTerminal).toBeGreaterThan(firstStarted)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true }, model: { completeResponse: 'worker summary' } }))))
})