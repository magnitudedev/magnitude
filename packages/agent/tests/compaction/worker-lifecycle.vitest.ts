import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { getCompaction, getTurn, mkContextLimitHit } from './helpers'

const workerLayer = TestHarnessLive({ workers: { compaction: true }, model: { completeResponse: 'worker summary' } })

const largeUserMessage = {
  type: 'user_message' as const,
  messageId: 'w',
  forkId: null,
  timestamp: Date.now(),
  content: [{ type: 'text' as const, text: 'X'.repeat(60_000) }],
  attachments: [],
  mode: 'text' as const,
  synthetic: false,
  taskMode: false,
}

describe.skip('compaction/worker-lifecycle', () => {
  it.effect('context_limit_hit triggers worker and emits compaction_ready', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ ...largeUserMessage, messageId: 'w1' })
      yield* h.send(mkContextLimitHit())
      const ready = yield* h.wait.event('compaction_ready', (e) => e.forkId === null)
      expect(ready.summary.length).toBeGreaterThan(0)
    }).pipe(Effect.provide(workerLayer)))

  it.effect('worker finalizes to compaction_completed and clears gates', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ ...largeUserMessage, messageId: 'w2' })
      yield* h.send(mkContextLimitHit())
      const completed = yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      expect(completed.tokensSaved).toBeGreaterThanOrEqual(0)
      const compaction2 = yield* getCompaction(h)
      expect(compaction2._tag).toBe('idle')
      expect(compaction2.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(workerLayer)))

  it.effect('worker failure emits compaction_failed and clears lifecycle flags', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ ...largeUserMessage, messageId: 'w3' })
      yield* h.send(mkContextLimitHit())
      const failed = yield* h.wait.event('compaction_failed', (e) => e.forkId === null)
      expect(failed.error.length).toBeGreaterThan(0)
      const compaction = yield* getCompaction(h)
      expect(compaction._tag).toBe('idle')
      expect(compaction.contextLimitBlocked).toBe(false)
    }).pipe(Effect.provide(TestHarnessLive({
      workers: { compaction: true },
      model: { completeResponse: () => { throw new Error('forced compaction failure') } },
    }))))

  it.effect('idempotent trigger does not overlap cycles', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ ...largeUserMessage, messageId: 'w4' })
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkContextLimitHit())
      yield* h.send(mkContextLimitHit())
      yield* h.wait.event('compaction_completed', (e) => e.forkId === null)

      const rootEvents = h.events().filter((e) => e.forkId === null)
      const starts = rootEvents.reduce<number[]>((acc, e, i) => e.type === 'compaction_started' ? [...acc, i] : acc, [])
      const terminals = rootEvents.reduce<number[]>((acc, e, i) => (e.type === 'compaction_completed' || e.type === 'compaction_failed') ? [...acc, i] : acc, [])
      expect(starts.length).toBeGreaterThanOrEqual(1)
      expect(terminals.length).toBeGreaterThanOrEqual(1)

      // Verify no overlapping cycles: each start must be followed by a terminal before the next start
      for (let i = 0; i < starts.length; i++) {
        const nextStart = starts[i + 1] ?? Infinity
        const terminalAfterThisStart = terminals.find((t) => t > starts[i])
        expect(terminalAfterThisStart).toBeDefined()
        expect(terminalAfterThisStart!).toBeLessThan(nextStart)
      }
    }).pipe(Effect.provide(workerLayer)))

  it.effect('worker ordering emits compaction_ready before compaction_completed', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({ ...largeUserMessage, messageId: 'w5' })
      yield* h.send(mkContextLimitHit())
      const ready = yield* h.wait.event('compaction_ready', (e) => e.forkId === null)
      const completed = yield* h.wait.event('compaction_completed', (e) => e.forkId === null)
      const events = h.events()
      const readyIndex = events.findIndex((e) => e.type === 'compaction_ready' && e.forkId === null)
      const completedIndex = events.findIndex((e) => e.type === 'compaction_completed' && e.forkId === null)
      expect(ready.summary.length).toBeGreaterThan(0)
      expect(completed.summary).toBe(ready.summary)
      expect(readyIndex).toBeGreaterThanOrEqual(0)
      expect(completedIndex).toBeGreaterThan(readyIndex)
    }).pipe(Effect.provide(workerLayer)))
})