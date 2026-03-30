import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { TurnProjection } from '../../src/projections/turn'
import {
  assertNoTurnIdMismatch,
  assertWorkingStateAligned,
  eventsForFork,
  mkTurnCompletedFailure,
  mkTurnCompletedSuccess,
  mkTurnStarted,
} from './helpers'

describe('turn control invariants', () => {
  it.live('single turn start/completion keeps turn IDs aligned', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-1', chainId: 'c-1' }))
      yield* h.send(mkTurnCompletedSuccess({ turnId: 't-1', chainId: 'c-1' }))

      const events = eventsForFork(h, null)
      assertNoTurnIdMismatch(events)
      yield* assertWorkingStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multiple sequential turns preserve pairing', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      for (const n of [1, 2, 3]) {
        yield* h.send(mkTurnStarted({ turnId: `t-${n}`, chainId: 'c-seq' }))
        yield* h.send(mkTurnCompletedSuccess({ turnId: `t-${n}`, chainId: 'c-seq' }))
      }

      const events = eventsForFork(h, null)
      assertNoTurnIdMismatch(events)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('failed completion still matches active turn ID', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-fail', chainId: 'c-fail' }))
      yield* h.send(mkTurnCompletedFailure({ turnId: 't-fail', chainId: 'c-fail' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertWorkingStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('turn_unexpected_error preserves turn-control alignment invariants', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-err', chainId: 'c-err' }))
      yield* h.send({ type: 'turn_unexpected_error', forkId: null, turnId: 't-err', message: 'bad stream' })

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertWorkingStateAligned(h)

      const turn = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(turn.activeTurn === null || turn.activeTurn.turnId !== '').toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('root and subagent forks remain ID-consistent independently', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send({
        type: 'agent_created',
        forkId: 'fork-a',
        parentForkId: null,
        agentId: 'agent-a',
        name: 'builder',
        role: 'builder',
        context: '',
        mode: 'spawn',
        taskId: 'task-a',
        message: 'spawn',
      })
      yield* h.send(mkTurnStarted({ forkId: null, turnId: 'root-1', chainId: 'root-c' }))
      yield* h.send(mkTurnStarted({ forkId: 'fork-a', turnId: 'sub-1', chainId: 'sub-c' }))
      yield* h.send(mkTurnCompletedSuccess({ forkId: null, turnId: 'root-1', chainId: 'root-c' }))
      yield* h.send(mkTurnCompletedSuccess({ forkId: 'fork-a', turnId: 'sub-1', chainId: 'sub-c' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      assertNoTurnIdMismatch(eventsForFork(h, 'fork-a'), 'fork-a')
      yield* assertWorkingStateAligned(h, null)
      yield* assertWorkingStateAligned(h, 'fork-a')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
