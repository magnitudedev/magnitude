import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'

describe('interrupt cancellation latency (regression)', () => {
  it.live('interrupt should complete cancelled turn within 1000ms for hanging stream (RED)', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({
        xmlChunks: ['<magnitude:message to="user">hello', ' world</magnitude:message><magnitude:yield_user/>'],
        delayMsBetweenChunks: 10_000,
      })

      yield* h.user('trigger hanging turn')

      yield* h.wait.event('turn_started', undefined, { timeoutMs: 1000 })
      yield* h.send({ type: 'interrupt', forkId: null })

      const completed = yield* h.wait.event(
        'turn_outcome',
        (e) => e.forkId === null && e.outcome._tag === 'Cancelled',
        { timeoutMs: 1000 },
      )

      expect(completed.outcome._tag).toBe('Cancelled')

      const outcomes = h.events().filter((e) => e.type === 'turn_outcome' && e.forkId === null)
      expect(outcomes).toHaveLength(1)
      expect(outcomes[0]?.outcome._tag).toBe('Cancelled')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
