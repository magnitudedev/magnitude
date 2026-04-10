import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import type { AppEvent } from '../../src/events'
import { TurnProjection } from '../../src/projections/turn'
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

  it.live('root interrupt clears communication trigger, preserves buffered message, and later wake consumes it', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-int-msg-1', chainId: 'c-int-msg' }))
      yield* h.user('buffered after interrupt')

      const beforeInterrupt = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(beforeInterrupt.pendingInboundCommunications).toHaveLength(1)
      expect(beforeInterrupt.pendingInboundCommunications[0]?.content).toBe('buffered after interrupt')
      expect(beforeInterrupt.triggers).toEqual([{ _tag: 'communication' }])

      yield* h.send({ type: 'interrupt', forkId: null })

      const interrupting = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(interrupting._tag).toBe('interrupting')
      expect(interrupting.triggers).toEqual([])
      expect(interrupting.pendingInboundCommunications).toHaveLength(1)

      yield* h.send(mkTurnCompletedFailure({
        turnId: 't-int-msg-1',
        chainId: 'c-int-msg',
        result: { success: false, error: 'interrupted', cancelled: true },
      }))

      const afterCancel = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(afterCancel._tag).toBe('idle')
      expect(afterCancel.triggers).toEqual([])
      expect(afterCancel.pendingInboundCommunications).toHaveLength(1)
      expect(afterCancel.pendingInboundCommunications[0]?.content).toBe('buffered after interrupt')

      const turnStartedEventsAfterCancel = eventsForFork(h, null).filter((event) => event.type === 'turn_started')
      expect(turnStartedEventsAfterCancel.map((event) => event.turnId)).toEqual(['t-int-msg-1'])

      yield* h.send({ type: 'wake', forkId: null })
      const resumed = yield* h.wait.event('turn_started', (event) => event.forkId === null && event.turnId !== 't-int-msg-1')

      const activeAgain = yield* h.projectionFork(TurnProjection.Tag, null)
      if (activeAgain._tag !== 'active') {
        expect.fail(`expected resumed root fork to be active, got ${activeAgain._tag}`)
      }
      expect(activeAgain.turnId).toBe(resumed.turnId)
      expect(activeAgain.currentTurnAllowsDirectUserReply).toBe(true)
      expect(activeAgain.pendingInboundCommunications).toHaveLength(0)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
