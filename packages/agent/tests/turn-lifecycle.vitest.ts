import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'

describe('turn lifecycle', () => {
  it.live('Single turn with yield', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '<message to="user">hi</message><idle/>' })

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
      yield* h.script.next({ xml: '<message to="user">first</message><idle/>' })

      yield* h.user('run')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Multi-turn conversation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({ xml: '<message to="user">response 1</message><idle/>' })
      yield* h.user('message 1')
      const first = yield* h.wait.turnCompleted(null)

      yield* h.script.next({ xml: '<message to="user">response 2</message><idle/>' })
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

  it.live('Empty response retriggers root turn instead of idling', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '' })
      yield* h.script.next({ xml: '<idle/>' })

      yield* h.user('trigger empty response')
      const first = yield* h.wait.turnCompleted(null)
      const second = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.result.success).toBe(true)
      if (first.result.success) {
        expect(first.result.turnDecision).toBe('continue')
      }

      expect(second.result.success).toBe(true)
      if (second.result.success) {
        expect(second.result.turnDecision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
