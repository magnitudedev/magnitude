import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { CompactionProjection } from '../../src/projections/compaction'
import {
  assertNoTurnIdMismatch,
  eventsForFork,
  mkContextLimitHit,
  mkTurnCompletedFailure,
  mkTurnCompletedSuccess,
  mkTurnStarted,
} from './helpers'

describe('turn control compaction gating interaction', () => {
  it.live('context_limit_hit blocks triggering while gate is active', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-g1', chainId: 'c-g' }))
      yield* h.send(mkContextLimitHit())

      const blockedDuring = yield* h.projectionFork(CompactionProjection.Tag, null)
      expect(blockedDuring.contextLimitBlocked).toBe(true)

      yield* h.send(mkTurnCompletedFailure({ turnId: 't-g1', chainId: 'c-g' }))

      const startedBefore = eventsForFork(h, null).filter((e) => e.type === 'turn_started').length
      yield* h.send({
        type: 'user_message',
        messageId: 'gated-msg',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'should be gated' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })
      const startedAfter = eventsForFork(h, null).filter((e) => e.type === 'turn_started').length
      expect(startedAfter).toBe(startedBefore)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )

  it.live('compaction_ready pending finalization keeps triggering gated', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({ type: 'compaction_started', forkId: null, compactedMessageCount: 4 })
      yield* h.send({
        type: 'compaction_ready',
        forkId: null,
        summary: 'summary',
        compactedMessageCount: 4,
        originalTokenEstimate: 5000,
        refreshedContext: null,
      })

      const compactionState = yield* h.projectionFork(CompactionProjection.Tag, null)
      expect(compactionState._tag !== 'idle').toBe(true)

      const startedBefore = eventsForFork(h, null).filter((e) => e.type === 'turn_started').length
      yield* h.send({
        type: 'user_message',
        messageId: 'gated-msg-2',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'still gated while pending' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })
      const startedAfter = eventsForFork(h, null).filter((e) => e.type === 'turn_started').length
      expect(startedAfter).toBe(startedBefore)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )

  it.live('unblock transition allows exactly one next turn with fresh ID', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-g2-old', chainId: 'c-g2' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnCompletedFailure({ turnId: 't-g2-old', chainId: 'c-g2' }))
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'done',
        compactedMessageCount: 3,
        tokensSaved: 1000,
        preservedVariables: [],
        refreshedContext: null,
      })

      yield* h.send(mkTurnStarted({ turnId: 't-g2-new', chainId: 'c-g2' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-g2-new', chainId: 'c-g2' }))

      const starts = eventsForFork(h, null).filter((e) => e.type === 'turn_started')
      expect(starts.filter((s) => s.turnId === 't-g2-new')).toHaveLength(1)
      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )

  it.live('completion around gate transitions remains mapped to active turn', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-g3', chainId: 'c-g3' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnCompletedFailure({ turnId: 't-g3', chainId: 'c-g3' }))
      yield* h.send({ type: 'compaction_started', forkId: null, compactedMessageCount: 1 })
      yield* h.send({
        type: 'compaction_ready',
        forkId: null,
        summary: 's',
        compactedMessageCount: 1,
        originalTokenEstimate: 1234,
        refreshedContext: null,
      })
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 's',
        compactedMessageCount: 1,
        tokensSaved: 900,
        preservedVariables: [],
        refreshedContext: null,
      })

      yield* h.send(mkTurnStarted({ turnId: 't-g3-next', chainId: 'c-g3' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-g3-next', chainId: 'c-g3' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )

  it.live('end-to-end blocked/pending/unblock sequence preserves global invariant', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-g4-1', chainId: 'c-g4' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkTurnCompletedFailure({ turnId: 't-g4-1', chainId: 'c-g4' }))
      yield* h.send({ type: 'compaction_started', forkId: null, compactedMessageCount: 2 })
      yield* h.send({
        type: 'compaction_ready',
        forkId: null,
        summary: 'sum',
        compactedMessageCount: 2,
        originalTokenEstimate: 6000,
        refreshedContext: null,
      })
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'sum',
        compactedMessageCount: 2,
        tokensSaved: 2000,
        preservedVariables: [],
        refreshedContext: null,
      })
      yield* h.send(mkTurnStarted({ turnId: 't-g4-2', chainId: 'c-g4' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-g4-2', chainId: 'c-g4' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive({ workers: { compaction: true } })))
  )
})
