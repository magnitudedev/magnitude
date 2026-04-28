import { describe, it } from '@effect/vitest'
import { expect } from 'vitest'
import { Effect } from 'effect'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import { TurnProjection } from '../../src/projections/turn'
import {
  assertNoTurnIdMismatch,
  assertTurnStateAligned,
  eventsForFork,
  mkTurnOutcomeEventFailure,
  mkTurnOutcomeEventSuccess,
  mkTurnStarted,
} from './helpers'

describe('turn control invariants', () => {
  it.live('single turn start/completion keeps turn IDs aligned', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-1', chainId: 'c-1' }))
      yield* h.send(mkTurnOutcomeEventSuccess({ turnId: 't-1', chainId: 'c-1' }))

      const events = eventsForFork(h, null)
      assertNoTurnIdMismatch(events)
      yield* assertTurnStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('multiple sequential turns preserve pairing', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      for (const n of [1, 2, 3]) {
        yield* h.send(mkTurnStarted({ turnId: `t-${n}`, chainId: 'c-seq' }))
        yield* h.send(mkTurnOutcomeEventSuccess({ turnId: `t-${n}`, chainId: 'c-seq' }))
      }

      const events = eventsForFork(h, null)
      assertNoTurnIdMismatch(events)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('failed completion still matches active turn ID', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-fail', chainId: 'c-fail' }))
      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-fail', chainId: 'c-fail' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertTurnStateAligned(h)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('turn_outcome preserves turn-control alignment invariants', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness
      yield* h.send(mkTurnStarted({ turnId: 't-err', chainId: 'c-err' }))
      yield* h.send(mkTurnOutcomeEventFailure({ turnId: 't-err', chainId: 'c-err' }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertTurnStateAligned(h)

      const turn = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(turn._tag === 'idle' || turn.turnId.length > 0).toBe(true)
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

      const subStart = eventsForFork(h, 'fork-a').find(
        (e): e is Extract<typeof e, { type: 'turn_started' }> => e.type === 'turn_started',
      )

      const subTurnId = subStart?.turnId ?? 'sub-1'
      const subChainId = subStart?.chainId ?? 'sub-c'

      if (!subStart) {
        yield* h.send(mkTurnStarted({ forkId: 'fork-a', turnId: subTurnId, chainId: subChainId }))
      }

      yield* h.send(mkTurnOutcomeEventSuccess({ forkId: null, turnId: 'root-1', chainId: 'root-c' }))
      yield* h.send(mkTurnOutcomeEventSuccess({ forkId: 'fork-a', turnId: subTurnId, chainId: subChainId }))

      assertNoTurnIdMismatch(eventsForFork(h, null))
      yield* assertTurnStateAligned(h, null)
      yield* assertTurnStateAligned(h, 'fork-a')
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
