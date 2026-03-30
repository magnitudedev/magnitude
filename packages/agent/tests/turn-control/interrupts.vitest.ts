import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import type { AppEvent } from '../../src/events'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import {
  assertNoTurnIdMismatch,
  assertTurnStateAligned,
  eventsForFork,
  mkContextLimitHit,
  mkTurnCompletedFailure,
  mkTurnCompletedSuccess,
  mkTurnStarted,
} from './helpers'

describe('turn control interrupts', () => {
  it.live('interrupt clears current turn state', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-int-1', chainId: 'c-int' }))
      yield* h.send({ type: 'interrupt', forkId: null })

      yield* assertTurnStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('completion after interrupt keeps interrupted turnId alignment', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-int-2', chainId: 'c-int' }))
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.send(mkTurnCompletedFailure({ turnId: 't-int-2', chainId: 'c-int' }))

      const events = eventsForFork(h, null)
      const started = events.find((e): e is Extract<AppEvent, { type: 'turn_started' }> => e.type === 'turn_started' && e.turnId === 't-int-2')
      const completed = events.find((e): e is Extract<AppEvent, { type: 'turn_completed' }> => e.type === 'turn_completed' && e.turnId === 't-int-2')
      expect(started).toBeDefined()
      expect(completed).toBeDefined()
      expect(completed!.turnId).toBe(started!.turnId)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('interrupt during context-limit blocked phase avoids mismatched future completion', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-int-3', chainId: 'c-int-3' }))
      yield* h.send(mkContextLimitHit())
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.send(mkTurnStarted({ turnId: 't-int-4', chainId: 'c-int-4' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-int-4', chainId: 'c-int-4' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('soft interrupt preserves active turn ID tracking until completion', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ forkId: 'fork-soft', turnId: 'soft-1', chainId: 'soft-c' }))
      yield* h.send({ type: 'soft_interrupt', forkId: 'fork-soft' })
      yield* h.send(mkTurnCompletedSuccess({ forkId: 'fork-soft', turnId: 'soft-1', chainId: 'soft-c' }))

      assertNoTurnIdMismatch(eventsForFork(h, 'fork-soft'))
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
