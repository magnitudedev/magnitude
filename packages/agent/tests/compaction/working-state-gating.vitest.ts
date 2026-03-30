import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { shouldTrigger } from '../../src/projections/working-state'
import {
  assertShouldTriggerBlocked,
  expectCompactionUnblocked,
  expectStableWorkingState,
  getCompaction,
  getWorking,
  mkCompactionFailed,
  mkCompactionReady,
  mkCompactionStarted,
  mkContextLimitHit,
  mkInterrupt,
  mkUserMessage,
  startReadyCompaction,
} from './helpers'

describe('compaction/working-state-gating', () => {
  it.effect('compaction_ready sets compactionPending and blocks shouldTrigger', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkUserMessage({ text: 'wake up' }))
      yield* h.send(mkCompactionReady())
      const state = yield* getWorking(h)
      expect(state.compactionPending).toBe(true)
      expect(shouldTrigger(state)).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit sets contextLimitBlocked and blocks shouldTrigger', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkUserMessage({ text: 'wake up' }))
      yield* h.send(mkContextLimitHit())
      const state = yield* getWorking(h)
      expect(state.contextLimitBlocked).toBe(true)
      expect(shouldTrigger(state)).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_completed clears both gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkCompactionReady())
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'summary',
        compactedMessageCount: 1,
        tokensSaved: 5,
        preservedVariables: [],
        refreshedContext: null,
      })
      yield* expectCompactionUnblocked(h)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('compaction_failed clears contextLimitBlocked but leaves compactionPending true', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      const state = yield* getWorking(h)
      expect(state.contextLimitBlocked).toBe(false)
      expect(state.compactionPending).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('eventual unblock invariant: completed path clears gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* startReadyCompaction(h)
      yield* h.send({
        type: 'compaction_completed',
        forkId: null,
        summary: 'summary',
        compactedMessageCount: 1,
        tokensSaved: 5,
        preservedVariables: [],
        refreshedContext: null,
      })
      const compaction = yield* getCompaction(h)
      const working = yield* getWorking(h)
      expect(compaction.contextLimitBlocked).toBe(false)
      expect(working.contextLimitBlocked).toBe(false)
      expect(working.compactionPending).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('eventual unblock invariant: failed + interrupt path clears gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkCompactionReady())
      yield* h.send(mkCompactionFailed())
      yield* h.send(mkInterrupt())
      yield* expectCompactionUnblocked(h)
      yield* expectStableWorkingState(h)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('eventual unblock invariant: interrupt during pending finalization clears gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkCompactionStarted())
      yield* h.send(mkCompactionReady())
      yield* assertShouldTriggerBlocked(h)
      yield* h.send(mkInterrupt())
      yield* expectCompactionUnblocked(h)
    }).pipe(Effect.provide(TestHarnessLive())))
})