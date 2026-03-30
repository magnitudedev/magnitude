import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getWorking, mkCompactionReady, mkCompactionStarted, mkContextLimitHit, mkInterrupt, mkCompactionFailed } from './helpers'

describe('compaction/interrupt', () => {
  it.effect('interrupt during summarization phase clears working-state gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkInterrupt())
      const working = yield* getWorking(h)
      expect(working.compactionPending).toBe(false)
      expect(working.contextLimitBlocked).toBe(false)
      expect(working.working).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt during pending finalization clears compactionPending', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionReady())
      yield* h.send(mkInterrupt())
      const working = yield* getWorking(h)
      expect(working.compactionPending).toBe(false)
      expect(working.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt while contextLimitBlocked clears block', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkInterrupt())
      const working = yield* getWorking(h)
      expect(working.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('interrupt after compaction_failed recovers residual pending gate', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      yield* h.send(mkInterrupt())
      const working = yield* getWorking(h)
      expect(working.compactionPending).toBe(false)
      expect(working.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))
})