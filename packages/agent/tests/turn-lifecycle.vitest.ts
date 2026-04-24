import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { YIELD_USER, YIELD_INVOKE } from '@magnitudedev/xml-act'
import { TestHarness, TestHarnessLive } from '../src/test-harness/harness'
import { IDENTICAL_RESPONSE_BREAKER_THRESHOLD } from '../src/execution/execution-manager'

describe('turn lifecycle', () => {
  it.live('Single turn with yield', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: `<magnitude:message to="user">hi</magnitude:message>\n${YIELD_USER}` })

      yield* h.user('hello')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result._tag).toBe('Completed')
      if (completed.result._tag === 'Completed') {
        expect(completed.result.completion.decision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Single turn with next', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: `<magnitude:message to="user">first</magnitude:message>\n${YIELD_USER}` })

      yield* h.user('run')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_completed')
      expect(completed.result._tag).toBe('Completed')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Multi-turn conversation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({ xml: `<magnitude:message to="user">response 1</magnitude:message>\n${YIELD_USER}` })
      yield* h.user('message 1')
      const first = yield* h.wait.turnCompleted(null)

      yield* h.script.next({ xml: `<magnitude:message to="user">response 2</magnitude:message>\n${YIELD_USER}` })
      yield* h.user('message 2')
      const second = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.type).toBe('turn_completed')
      expect(second.type).toBe('turn_completed')
      expect(first.result._tag).toBe('Completed')
      expect(second.result._tag).toBe('Completed')
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
      expect(completed.result._tag).toBe('Completed')
      expect(chunk.text).toContain('ok')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Empty response retriggers root turn instead of idling', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: '' })
      yield* h.script.next({ xml: YIELD_USER })

      yield* h.user('trigger empty response')
      const first = yield* h.wait.turnCompleted(null)
      const second = yield* h.wait.event(
        'turn_completed',
        (e) => e.forkId === null && e.turnId !== first.turnId,
      )

      expect(first.result._tag).toBe('Completed')
      if (first.result._tag === 'Completed') {
        expect(first.result.completion.decision).toBe('continue')
      }

      expect(second.result._tag).toBe('Completed')
      if (second.result._tag === 'Completed') {
        expect(second.result.completion.decision).toBe('idle')
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

      yield* h.script.next({ xml: `<magnitude:message to="${taskId}">hello</magnitude:message>\n${YIELD_USER}` })

      yield* h.user('trigger workerless task message')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.result._tag).toBe('Completed')
      if (completed.result._tag === 'Completed') {
        expect(completed.result.completion.decision).toBe('continue')
        expect(completed.result.completion.feedback).toEqual([
          {
            _tag: 'InvalidMessageDestination',
            destination: taskId,
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
      const repeated = `<magnitude:message to="user">repeat</magnitude:message>\n${YIELD_INVOKE}`
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
        expect(turns[i].result._tag).toBe('Completed')
        if (turns[i].result._tag === 'Completed') {
          expect(turns[i].result.completion.decision).toBe('continue')
        }
      }

      // Last should trip
      const last = turns[N - 1]
      expect(last.result._tag).toBe('SystemError')
      if (last.result._tag === 'SystemError') {
        expect(last.result.message).toContain('Circuit breaker')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Circuit breaker resets on different response: A, A, B, A does not trip', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      const a = `<magnitude:message to="user">foo</magnitude:message>\n${YIELD_INVOKE}`
      const b = `<magnitude:message to="user">bar</magnitude:message>\n${YIELD_INVOKE}`
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: b })
      yield* h.script.next({ xml: a })
      yield* h.script.next({ xml: `<magnitude:message to="user">done</magnitude:message>\n${YIELD_USER}` })

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
        expect(turn.result._tag).toBe('Completed')
        if (turn.result._tag === 'Completed') {
          expect(turn.result.completion.decision).toBe('continue')
        }
      }

      expect(fifth.result._tag).toBe('Completed')
      if (fifth.result._tag === 'Completed') {
        expect(fifth.result.completion.decision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Circuit breaker resets on idle boundary', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      const a = `<magnitude:message to="user">foo</magnitude:message>\n${YIELD_INVOKE}`
      const idleWithMessage = `<magnitude:message to="user">boundary</magnitude:message>\n${YIELD_USER}`
      yield* h.script.next({ xml: a })                 // continue
      yield* h.script.next({ xml: a })                 // continue
      yield* h.script.next({ xml: idleWithMessage })   // idle boundary (reset)
      yield* h.script.next({ xml: a })                 // continue after reset
      yield* h.script.next({ xml: YIELD_USER })              // stop

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

      expect(first.result._tag).toBe('Completed')
      if (first.result._tag === 'Completed') {
        expect(first.result.completion.decision).toBe('continue')
      }

      expect(second.result._tag).toBe('Completed')
      if (second.result._tag === 'Completed') {
        expect(second.result.completion.decision).toBe('continue')
      }

      expect(third.result._tag).toBe('Completed')
      if (third.result._tag === 'Completed') {
        expect(third.result.completion.decision).toBe('idle')
      }

      expect(fourth.result._tag).toBe('Completed')
      if (fourth.result._tag === 'Completed') {
        expect(fourth.result.completion.decision).toBe('continue')
      }

      expect(fifth.result._tag).toBe('Completed')
      if (fifth.result._tag === 'Completed') {
        expect(fifth.result.completion.decision).toBe('idle')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
