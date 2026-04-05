import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'

describe('interrupt cancellation latency (regression)', () => {
  it.live('interrupt should complete cancelled turn within 1000ms for hanging stream (RED)', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({
        xmlChunks: ['<comms><message to="user">hello', ' world</message></comms><idle/>'],
        delayMsBetweenChunks: 10_000,
      })

      yield* h.user('trigger hanging turn')

      yield* h.wait.event('turn_started', undefined, { timeoutMs: 1000 })
      yield* h.send({ type: 'interrupt', forkId: null })

      const completed = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.result.success === false && !e.result.success && e.result.cancelled === true,
        { timeoutMs: 1000 },
      )

      expect(completed.result.success).toBe(false)
      if (!completed.result.success) {
        expect(completed.result.cancelled).toBe(true)
      }

      const unexpected = h.events().find((e) => e.type === 'turn_unexpected_error' && e.forkId === null)
      expect(unexpected).toBeUndefined()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
