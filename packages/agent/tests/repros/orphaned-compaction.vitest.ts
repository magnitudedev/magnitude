import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getCompaction, getTurn, mkContextLimitHit } from '../compaction/helpers'

/**
 * Reproduces the orphaned compaction bug found in session mnb0dvn5.
 *
 * In production, compaction_started fired at index 149101, then an interrupt
 * arrived at index 149102. No compaction_ready or compaction_completed ever
 * followed — the compaction cycle was orphaned.
 *
 * Exact sequence from production:
 *   149100: turn_started(w5dsuyqz7q1q)
 *   149101: compaction_started
 *   149102: interrupt
 *   149103: turn_outcome(w5dsuyqz7q1q)
 *   ... turns continue, but compaction never recovers
 *
 * The compaction worker's async summarization fiber should be cleaned up
 * on interrupt, and the system should be able to compact again later.
 */
describe('orphaned compaction after interrupt (mnb0dvn5 reproduction)', () => {
  const workerLayer = TestHarnessLive({
    workers: { compaction: true },
    model: { completeResponse: 'compaction summary' },
  })

  const largeUserMessage = {
    type: 'user_message' as const,
    messageId: 'orphan-msg',
    forkId: null,
    timestamp: Date.now(),
    content: [{ type: 'text' as const, text: 'X'.repeat(60_000) }],
    attachments: [] as never[],
    mode: 'text' as const,
    synthetic: false,
    taskMode: false,
  }

  it.live('interrupt during compaction does not orphan the cycle — worker recovers', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Seed large message to make compaction viable
      yield* h.send(largeUserMessage)

      // Trigger compaction via context_limit_hit
      yield* h.send(mkContextLimitHit())

      // Wait for compaction to start
      yield* h.wait.event('compaction_started', (e) => e.forkId === null)

      // Interrupt immediately (reproducing the production sequence)
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.wait.event('turn_outcome', (e) => e.forkId === null)

      const turn = yield* getTurn(h)
      expect(turn._tag).toBe('idle')

      // Verify compaction projection is not stuck
      const compaction = yield* getCompaction(h)
      expect(compaction._tag).toBe('idle')
      expect(compaction.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(workerLayer))
  )

  it.live('after interrupted compaction, a new compaction cycle can complete', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Seed large message
      yield* h.send(largeUserMessage)

      // First compaction: trigger and interrupt
      yield* h.send(mkContextLimitHit())
      yield* h.wait.event('compaction_started', (e) => e.forkId === null)
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.wait.event('turn_outcome', (e) => e.forkId === null)

      // Second compaction: trigger again — should work
      yield* h.send({ ...largeUserMessage, messageId: 'orphan-msg-2' })
      yield* h.send(mkContextLimitHit())

      // The worker should be able to start a new compaction cycle
      yield* h.wait.event('compaction_started', (e) => e.forkId === null)

      // And it should complete successfully
      const completed = yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      expect(completed.summary.length).toBeGreaterThan(0)

      // Final state should be clean
      const compaction = yield* getCompaction(h)
      expect(compaction._tag).toBe('idle')
      expect(compaction.contextLimitBlocked).toBe(false)

      const turn = yield* getTurn(h)
      expect(turn._tag).toBe('idle')
    }).pipe(Effect.provide(workerLayer))
  )
})
