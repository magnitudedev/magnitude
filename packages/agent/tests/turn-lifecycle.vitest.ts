import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from '../src/execution/execution-manager'

describe('turn lifecycle', () => {
  it.live('Single turn with yield', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: `<message to="user">hi</message>\n${TURN_CONTROL_IDLE}` })

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
      yield* h.script.next({ xml: `<message to="user">first</message>\n${TURN_CONTROL_IDLE}` })

      yield* h.user('run')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Multi-turn conversation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({ xml: `<message to="user">response 1</message>\n${TURN_CONTROL_IDLE}` })
      yield* h.user('message 1')
      const first = yield* h.wait.turnCompleted(null)

      yield* h.script.next({ xml: `<message to="user">response 2</message>\n${TURN_CONTROL_IDLE}` })
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
      const chunk = yield* h.wait.event(
        'message_chunk',
        (e) => e.forkId === null && e.turnId === completed.turnId,
      )

      expect(completed.type).toBe('turn_completed')
      expect(completed.result.success).toBe(true)
      expect(chunk.text).toContain('ok')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Empty response retriggers root turn instead of idling', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '' })
      yield* h.script.next({ xml: TURN_CONTROL_IDLE })

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

  it.live('message to task without spawned worker surfaces destination error', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      const taskId = 'task-no-worker'

      yield* h.send({
        type: 'task_created',
        forkId: null,
        taskId,
        title: 'Task without worker',
        parentId: null,
        timestamp: Date.now(),
      })

      yield* h.script.next({ xml: `<message to="${taskId}">hello</message>\n${TURN_CONTROL_IDLE}` })

      yield* h.user('trigger workerless task message')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.result.success).toBe(true)
      if (completed.result.success) {
        expect(completed.result.turnDecision).toBe('continue')
        expect(completed.result.errors).toEqual([
          {
            code: 'nonexistent_agent_destination',
            message: `Invalid message destination "${taskId}": task has no active worker`,
          },
        ])
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live(`Circuit breaker trips after N identical consecutive continue responses`, () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      const N = IDENTICAL_RESPONSE_BREAKER_THRESHOLD
      // Invalid tool call body forces turn continue with tool parse error.
      const repeated = '<read>foo</read>'
      for (let i = 0; i < N; i++) {
        yield* h.script.next({ xml: repeated })
      }

      yield* h.user('trigger repeated identical invalid response loop')

      const turnIds: string[] = []
      const turns: any[] = []
      for (let i = 0; i < N; i++) {
        const turn = yield* h.wait.event(
          'turn_completed',
          (e) => e.forkId === null && !turnIds.includes(e.turnId),
        )
        turnIds.push(turn.turnId)
        turns.push(turn)
      }

      // All but last should continue
      for (let i = 0; i < N - 1; i++) {
        expect(turns[i].result.success).toBe(true)
        if (turns[i].result.success) {
          expect(turns[i].result.turnDecision).toBe('continue')
        }
      }

      // Last should trip
      const last = turns[N - 1]
      expect(last.result.success).toBe(false)
      if (!last.result.success) {
        expect(last.result.error).toContain('Circuit breaker')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Circuit breaker resets on different response: A, A, B, A does not trip', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      const a = '<read>foo</read>'
      const b = '<read>bar</read>'
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: b })
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: TURN_CONTROL_IDLE })

      yield* h.user('trigger reset on different response sequence')

      const first = yield* h.wait.turnCompleted(null)
      const second = yield* h.wait.event('turn_completed', (e) => e.forkId === null && e.turnId !== first.turnId)
      const third = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId && e.turnId !== second.turnId,
      )
      const fourth = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && ![first.turnId, second.turnId, third.turnId].includes(e.turnId),
      )
      const fifth = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && ![first.turnId, second.turnId, third.turnId, fourth.turnId].includes(e.turnId),
      )

      for (const turn of [first, second, third, fourth]) {
        expect(turn.result.success).toBe(true)
        if (turn.result.success) {
          expect(turn.result.turnDecision).toBe('continue')
        }
      }

      expect(fifth.result.success).toBe(true)
      if (fifth.result.success) {
        expect(fifth.result.turnDecision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Circuit breaker resets on idle boundary', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      const a = '<read>foo</read>'
      const idleWithMessage = `<message to="user">boundary</message>\n${TURN_CONTROL_IDLE}`
      yield* h.script.next({ xml: a })                 // continue
      yield* h.script.next({ xml: a })                 // continue
      yield* h.script.next({ xml: idleWithMessage })   // idle boundary (reset)
      yield* h.script.next({ xml: a })                 // continue after reset
      yield* h.script.next({ xml: TURN_CONTROL_IDLE })              // stop

      yield* h.user('trigger reset on idle boundary sequence')

      const first = yield* h.wait.turnCompleted(null)
      const second = yield* h.wait.event('turn_completed', (e) => e.forkId === null && e.turnId !== first.turnId)
      const third = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId && e.turnId !== second.turnId,
      )

      // After idle boundary, need a new user message to resume
      yield* h.user('continue after idle')

      const fourth = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && ![first.turnId, second.turnId, third.turnId].includes(e.turnId),
      )
      const fifth = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && ![first.turnId, second.turnId, third.turnId, fourth.turnId].includes(e.turnId),
      )

      expect(first.result.success).toBe(true)
      if (first.result.success) {
        expect(first.result.turnDecision).toBe('continue')
      }

      expect(second.result.success).toBe(true)
      if (second.result.success) {
        expect(second.result.turnDecision).toBe('continue')
      }

      expect(third.result.success).toBe(true)
      if (third.result.success) {
        expect(third.result.turnDecision).toBe('idle')
      }

      expect(fourth.result.success).toBe(true)
      if (fourth.result.success) {
        expect(fourth.result.turnDecision).toBe('continue')
      }

      expect(fifth.result.success).toBe(true)
      if (fifth.result.success) {
        expect(fifth.result.turnDecision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
