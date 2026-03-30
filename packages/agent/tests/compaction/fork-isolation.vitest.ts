import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'

import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { createSubagentFork, getTurn, mkCompactionCompleted, mkCompactionReady, mkCompactionStarted, mkContextLimitHit, mkUserMessage } from './helpers'

describe('compaction/fork-isolation', () => {
  it.effect('root compaction gates do not mutate subagent working state', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      const subBefore = yield* getTurn(h, subFork)
      yield* h.send(mkContextLimitHit(null))
      yield* h.send(mkCompactionStarted(null))
      yield* h.send(mkCompactionReady({ forkId: null }))
      yield* h.send(mkCompactionCompleted({ forkId: null }))
      const subAfter = yield* getTurn(h, subFork)
      expect(subAfter._tag).toBe(subBefore._tag)
      expect(subAfter.triggers.length).toBe(subBefore.triggers.length)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } }))))

  it.effect('subagent compaction gates do not mutate root working state', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      const rootBefore = yield* getTurn(h, null)
      yield* h.send(mkContextLimitHit(subFork))
      yield* h.send(mkCompactionStarted(subFork))
      yield* h.send(mkCompactionReady({ forkId: subFork }))
      yield* h.send(mkCompactionCompleted({ forkId: subFork }))
      const rootAfter = yield* getTurn(h, null)
      expect(rootAfter._tag).toBe(rootBefore._tag)
      expect(rootAfter.triggers.length).toBe(rootBefore.triggers.length)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } }))))

  it.effect('independent root and subagent cycles clear own flags only', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      yield* h.send(mkCompactionStarted(null))
      yield* h.send(mkCompactionReady({ forkId: null }))
      yield* h.send(mkCompactionStarted(subFork))
      yield* h.send(mkCompactionReady({ forkId: subFork }))
      yield* h.send(mkCompactionCompleted({ forkId: subFork }))
      const rootMid = yield* getTurn(h, null)
      expect(rootMid._tag).toBe('idle')
      yield* h.send(mkCompactionCompleted({ forkId: null }))
      const rootAfter = yield* getTurn(h, null)
      const subAfter = yield* getTurn(h, subFork)
      expect(rootAfter._tag).toBe('idle')
      expect(subAfter._tag).toBe('idle')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } }))))

  it.effect('context_limit_hit on root does not block sibling shouldTrigger', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const subFork = yield* createSubagentFork(h)
      yield* h.send({ type: 'wake', forkId: subFork })
      const before = yield* getTurn(h, subFork)
      yield* h.send(mkContextLimitHit(null))
      const after = yield* getTurn(h, subFork)
      expect(after._tag === 'idle' && after.triggers.length > 0).toBe(before._tag === 'idle' && before.triggers.length > 0)
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } }))))
})