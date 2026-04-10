import { describe, expect, it } from '@effect/vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../harness'

describe('fault injection', () => {
  it.live('Malformed XML', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next({ xml: '<message to="user">hi' }, null)

      yield* harness.user('broken xml')

      const result = yield* Effect.race(
        harness.wait.turnCompleted(null),
        harness.wait.event('turn_unexpected_error'),
      )

      expect(result.type === 'turn_completed' || result.type === 'turn_unexpected_error').toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('terminateStreamEarly', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next(
        {
          xmlChunks: ['', '<message to="user">hi</message>', '<idle/>'],
          terminateStreamEarly: true,
        },
        null,
      )

      yield* harness.user('terminate early')

      const result = yield* Effect.race(
        harness.wait.turnCompleted(null),
        harness.wait.event('turn_unexpected_error'),
      )

      expect(result.type === 'turn_completed' || result.type === 'turn_unexpected_error').toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('failAfterChunk', () =>
    Effect.gen(function* () {
      const harness = yield* TestHarness
      yield* harness.script.next(
        {
          xmlChunks: ['', '<message to="user">hi</message>', '<idle/>'],
          failAfterChunk: 1,
        },
        null,
      )

      yield* harness.user('fail after chunk')

      const result = yield* Effect.race(
        harness.wait.event('turn_unexpected_error'),
        harness.wait.turnCompleted(null),
      )

      expect(result.type === 'turn_unexpected_error' || result.type === 'turn_completed').toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
