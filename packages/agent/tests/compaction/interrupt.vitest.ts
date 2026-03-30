import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getCompaction, getTurn, mkCompactionReady, mkCompactionStarted, mkContextLimitHit, mkInterrupt, mkCompactionFailed } from './helpers'

describe('compaction/interrupt', () => {
  it.effect('interrupt during summarization phase clears working-state gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkInterrupt())
      const compactionState = yield* getCompaction(h)
      const turnState = yield* getTurn(h)
      expect(compactionState._tag !== 'idle').toBe(false)
      expect(compactionState.contextLimitBlocked).toBe(false)
      expect(turnState._tag !== 'idle').toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt during pending finalization clears compactionPending', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionReady())
      yield* h.send(mkInterrupt())
      const compactionState = yield* getCompaction(h)
      expect(compactionState._tag !== 'idle').toBe(false)
      expect(compactionState.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt while contextLimitBlocked clears block', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkInterrupt())
      const compactionState = yield* getCompaction(h)
      expect(compactionState.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt after compaction_failed recovers residual pending gate', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      yield* h.send(mkInterrupt())
      const compactionState = yield* getCompaction(h)
      expect(compactionState._tag !== 'idle').toBe(false)
      expect(compactionState.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))
})