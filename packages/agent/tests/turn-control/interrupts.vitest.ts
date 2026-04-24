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

const mkUserMessage = (id: string, text: string): Extract<AppEvent, { type: 'user_message' }> => ({
  type: 'user_message',
  messageId: id,
  forkId: null,
  timestamp: Date.now(),
  content: [{ type: 'text', text }],
  attachments: [],
  mode: 'text',
  synthetic: false,
  taskMode: false,
})

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

  it.live('fresh user message after interrupted completion starts a new root turn without manual wake', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-fresh-1', chainId: 'c-fresh-1' }))
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.send(mkTurnCompletedFailure({
        turnId: 't-fresh-1',
        chainId: 'c-fresh-1',
        result: { _tag: 'Cancelled' },
      }))

      yield* h.send(mkUserMessage('msg-fresh-after-cancel', 'new instruction after cancel'))

      const resumed = yield* h.wait.event(
        'turn_started',
        (event) => event.forkId === null && event.turnId !== 't-fresh-1',
      )

      const activeAgain = yield* h.projectionFork(TurnProjection.Tag, null)
      if (activeAgain._tag !== 'active') {
        expect.fail(`expected fresh root turn to be active, got ${activeAgain._tag}`)
      }

      expect(activeAgain.turnId).toBe(resumed.turnId)
      expect(activeAgain.triggeredByUser).toBe(true)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('user message received during interrupt is accepted and does not wedge later root progress', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-buffer-1', chainId: 'c-buffer-1' }))
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.send(mkUserMessage('msg-buffered', 'resume with this'))

      const stillInterrupting = yield* h.projectionFork(TurnProjection.Tag, null)
      expect(stillInterrupting._tag).toBe('interrupting')
      expect(stillInterrupting.pendingInboundCommunications).toHaveLength(1)
      expect(stillInterrupting.pendingInboundCommunications[0]?.content).toBe('resume with this')

      yield* h.send(mkTurnCompletedFailure({
        turnId: 't-buffer-1',
        chainId: 'c-buffer-1',
        result: { _tag: 'Cancelled' },
      }))

      const nextTurn = yield* h.wait.event(
        'turn_started',
        (event) => event.forkId === null && event.turnId !== 't-buffer-1',
      )

      const root = yield* h.projectionFork(TurnProjection.Tag, null)
      if (root._tag !== 'active') {
        expect.fail(`expected root turn to be active after interrupted completion, got ${root._tag}`)
      }

      expect(root.turnId).toBe(nextTurn.turnId)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('stale completion from interrupted root turn does not block a later fresh user turn', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send(mkTurnStarted({ turnId: 't-stale-1', chainId: 'c-stale-1' }))
      yield* h.send({ type: 'interrupt', forkId: null })
      yield* h.send(mkTurnCompletedFailure({
        turnId: 't-stale-1',
        chainId: 'c-stale-1',
        result: { _tag: 'Cancelled' },
      }))

      yield* h.send(mkTurnCompletedSuccess({
        turnId: 't-stale-1',
        chainId: 'c-stale-1',
      }))

      yield* h.send(mkUserMessage('msg-stale-followup', 'follow-up instruction'))

      const followUp = yield* h.wait.event(
        'turn_started',
        (event) => event.forkId === null && event.turnId !== 't-stale-1',
      )

      const root = yield* h.projectionFork(TurnProjection.Tag, null)
      if (root._tag !== 'active') {
        expect.fail(`expected fresh follow-up root turn to remain active, got ${root._tag}`)
      }

      expect(root.turnId).toBe(followUp.turnId)
    }).pipe(Effect.provide(TestHarnessLive()))
  )

  it.live('subfork completion still wakes the parent after a root interrupt', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      yield* h.send({
        type: 'agent_created',
        forkId: 'sub-1',
        parentForkId: null,
        agentId: 'test-subagent',
        name: 'test-subagent',
        role: 'subagent',
        context: '',
        mode: 'spawn',
        taskId: 'test-task',
        message: '',
      })

      yield* h.send({ type: 'turn_started', forkId: 'sub-1', turnId: 'sub-turn-1', chainId: 'sub-chain-1' })
      yield* h.send(mkUserMessage('msg-parent', 'parent follow-up'))
      yield* h.send({ type: 'interrupt', forkId: null })

      yield* h.send(mkTurnCompletedSuccess({
        forkId: 'sub-1',
        turnId: 'sub-turn-1',
        chainId: 'sub-chain-1',
      }))

      const parentWake = yield* h.wait.event(
        'turn_started',
        (event) => event.forkId === null,
      )

      expect(parentWake.forkId).toBeNull()
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
