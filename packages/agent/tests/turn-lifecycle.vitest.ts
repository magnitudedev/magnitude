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

      expect(completed.type).toBe('turn_outcome')
      expect(completed.outcome._tag).toBe('Completed')
      if (completed.outcome._tag === 'Completed') {
        expect(completed.outcome.completion.yieldTarget).toBe('user')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Single turn with next', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.script.next({ xml: `<magnitude:message to="user">first</magnitude:message>\n${YIELD_USER}` })

      yield* h.user('run')
      const completed = yield* h.wait.turnCompleted(null)

      expect(completed.type).toBe('turn_outcome')
      expect(completed.outcome._tag).toBe('Completed')
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('Multi-turn conversation', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.script.next({ xml: `<magnitude:message to="user">response 1</magnitude:message>\n${YIELD_USER}` })
      yield* h.user('message 1')
      const first = yield* h.wait.turnCompleted(null)

      yield* h.script.next({ xml: `<magnitude:message to="user">response 2</magnitude:message>\n${YIELD_USER}` })
      const beforeSecond = h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length
      yield* h.user('message 2')
      yield* h.until('second root completion', () =>
        h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length > beforeSecond,
      )
      const second = h.events()
        .filter((e): e is Extract<typeof e, { type: 'turn_outcome' }> => e.type === 'turn_outcome' && e.forkId === null)
        .find(e => e.turnId !== first.turnId)
      if (!second) throw new Error('Expected second turn completion')

      expect(first.type).toBe('turn_outcome')
      expect(second.type).toBe('turn_outcome')
      expect(first.outcome._tag).toBe('Completed')
      expect(second.outcome._tag).toBe('Completed')
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

      expect(completed.type).toBe('turn_outcome')
      expect(completed.outcome._tag).toBe('Completed')
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
      yield* h.until('second root completion after retrigger', () =>
        h.events().some(e => e.type === 'turn_outcome' && e.forkId === null && e.turnId !== first.turnId),
      )
      const second = h.events().find((e): e is Extract<typeof e, { type: 'turn_outcome' }> =>
        e.type === 'turn_outcome' && e.forkId === null && e.turnId !== first.turnId,
      )
      if (!second) throw new Error('Expected second turn completion')

      expect(first.outcome._tag).toBe('Completed')
      if (first.outcome._tag === 'Completed') {
        expect(first.outcome.completion.yieldTarget).toBe('invoke')
      }

      expect(second.outcome._tag).toBe('Completed')
      if (second.outcome._tag === 'Completed') {
        expect(second.outcome.completion.yieldTarget).toBe('user')
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

      expect(completed.outcome._tag).toBe('Completed')
      if (completed.outcome._tag === 'Completed') {
        expect(completed.outcome.completion.yieldTarget).toBe('invoke')
        expect(completed.outcome.completion.feedback).toEqual([
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

      const turns: any[] = []
      yield* h.until('N root completions for circuit breaker', () =>
        h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length >= N,
      )
      turns.push(
        ...h.events().filter((e): e is Extract<typeof e, { type: 'turn_outcome' }> =>
          e.type === 'turn_outcome' && e.forkId === null,
        ).slice(0, N),
      )

      // All but last should continue
      for (let i = 0; i < N - 1; i++) {
        expect(turns[i].outcome._tag).toBe('Completed')
        if (turns[i].outcome._tag === 'Completed') {
          expect(turns[i].outcome.completion.yieldTarget).toBe('invoke')
        }
      }

      // Last should trip
      const last = turns[N - 1]
      expect(last.outcome._tag).toBe('SafetyStop')
      if (last.outcome._tag === 'SafetyStop') {
        expect(last.outcome.reason._tag).toBe('IdenticalResponseCircuitBreaker')
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
      yield* h.until('five root completions', () =>
        h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length >= 5,
      )
      const [first, second, third, fourth, fifth] = h.events().filter(
        (e): e is Extract<typeof e, { type: 'turn_outcome' }> => e.type === 'turn_outcome' && e.forkId === null,
      )

      for (const turn of [first, second, third, fourth]) {
        expect(turn.outcome._tag).toBe('Completed')
        if (turn.outcome._tag === 'Completed') {
          expect(turn.outcome.completion.yieldTarget).toBe('invoke')
        }
      }

      expect(fifth.outcome._tag).toBe('Completed')
      if (fifth.outcome._tag === 'Completed') {
        expect(fifth.outcome.completion.yieldTarget).toBe('user')
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
      yield* h.until('three root completions before idle boundary', () =>
        h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length >= 3,
      )
      const [, second, third] = h.events().filter(
        (e): e is Extract<typeof e, { type: 'turn_outcome' }> => e.type === 'turn_outcome' && e.forkId === null,
      )

      // After idle boundary, need a new user message to resume
      yield* h.user('continue after idle')

      yield* h.until('five root completions after resume', () =>
        h.events().filter(e => e.type === 'turn_outcome' && e.forkId === null).length >= 5,
      )
      const [, , , fourth, fifth] = h.events().filter(
        (e): e is Extract<typeof e, { type: 'turn_outcome' }> => e.type === 'turn_outcome' && e.forkId === null,
      )

      expect(first.outcome._tag).toBe('Completed')
      if (first.outcome._tag === 'Completed') {
        expect(first.outcome.completion.yieldTarget).toBe('invoke')
      }

      expect(second.outcome._tag).toBe('Completed')
      if (second.outcome._tag === 'Completed') {
        expect(second.outcome.completion.yieldTarget).toBe('invoke')
      }

      expect(third.outcome._tag).toBe('Completed')
      if (third.outcome._tag === 'Completed') {
        expect(third.outcome.completion.yieldTarget).toBe('user')
      }

      expect(fourth.outcome._tag).toBe('Completed')
      if (fourth.outcome._tag === 'Completed') {
        expect(fourth.outcome.completion.yieldTarget).toBe('invoke')
      }

      expect(fifth.outcome._tag).toBe('Completed')
      if (fifth.outcome._tag === 'Completed') {
        expect(fifth.outcome.completion.yieldTarget).toBe('user')
      }
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
