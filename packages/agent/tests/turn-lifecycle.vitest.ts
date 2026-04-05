import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'

describe('turn lifecycle', () => {
  it.live('Single turn with yield', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '<comms><message to="user">hi</message></comms><idle/>' })

      yield* h.user('hello')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
      if (completed.result.success) {
        expect(completed.result.turnDecision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Single turn with next', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '<comms><message to="user">first</message></comms><idle/>' })

      yield* h.user('run')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Multi-turn conversation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({ xml: '<comms><message to="user">response 1</message></comms><idle/>' })
      yield* h.user('message 1')
      const first = yield* h.wait.turnCompleted(null)

      yield* h.script.next({ xml: '<comms><message to="user">response 2</message></comms><idle/>' })
      yield* h.user('message 2')
      const second = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.type).toBe('turn_completed')
      expect(second.type).toBe('turn_completed')
      expect(first.result.success).toBe(true)
      expect(second.result.success).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Default frame when no script queued', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.user('no script queued')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)

      const text = completed.responseParts.find((p) => p.type === 'text')
      expect(text?.type).toBe('text')
      if (text?.type === 'text') {
        expect(text.content).toContain('ok')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
