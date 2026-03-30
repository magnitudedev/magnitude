import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { shouldTrigger } from '../../src/projections/working-state'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { createSubagentFork, getWorking, mkCompactionCompleted, mkCompactionReady, mkContextLimitHit, mkUserMessage } from './helpers'

describe('compaction/fork-isolation', () => {
  it.effect('root compaction gates do not mutate subagent working state', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      const subBefore = yield* getWorking(h, subFork)
      yield* h.send(mkContextLimitHit(null))
      yield* h.send(mkCompactionReady({ forkId: null }))
      yield* h.send(mkCompactionCompleted({ forkId: null }))
      const subAfter = yield* getWorking(h, subFork)
      expect(subAfter.contextLimitBlocked).toBe(subBefore.contextLimitBlocked)
      expect(subAfter.compactionPending).toBe(subBefore.compactionPending)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('subagent compaction gates do not mutate root working state', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      const rootBefore = yield* getWorking(h, null)
      yield* h.send(mkContextLimitHit(subFork))
      yield* h.send(mkCompactionReady({ forkId: subFork }))
      yield* h.send(mkCompactionCompleted({ forkId: subFork }))
      const rootAfter = yield* getWorking(h, null)
      expect(rootAfter.contextLimitBlocked).toBe(rootBefore.contextLimitBlocked)
      expect(rootAfter.compactionPending).toBe(rootBefore.compactionPending)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('independent root and subagent cycles clear own flags only', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      yield* h.send(mkCompactionReady({ forkId: null }))
      yield* h.send(mkCompactionReady({ forkId: subFork }))
      yield* h.send(mkCompactionCompleted({ forkId: subFork }))
      const rootMid = yield* getWorking(h, null)
      expect(rootMid.compactionPending).toBe(true)
      yield* h.send(mkCompactionCompleted({ forkId: null }))
      const rootAfter = yield* getWorking(h, null)
      const subAfter = yield* getWorking(h, subFork)
      expect(rootAfter.compactionPending).toBe(false)
      expect(subAfter.compactionPending).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive())))

  it.effect('context_limit_hit on root does not block sibling shouldTrigger', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      yield* h.send({ type: 'wake', forkId: subFork })
      const before = yield* getWorking(h, subFork)
      yield* h.send(mkContextLimitHit(null))
      const after = yield* getWorking(h, subFork)
      expect(shouldTrigger(after)).toBe(shouldTrigger(before))
    }).pipe(Effect.provide(TestHarnessLive())))
})