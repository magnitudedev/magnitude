import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'
import {
  assertNoTurnIdMismatch,
  eventsForFork,
  getTurn,
  mkTurnCompletedSuccess,
  mkTurnStarted,
} from '../turn-control/helpers'

/**
 * Reproduces the stale turn_completed bug found in session mnb0dvn5.
 *
 * Root cause: When a user interrupts and sends a new message, the turn
 * controller starts a new turn BEFORE the interrupted turn's completion
 * event is processed. The old completion arrives with a turnId that no
 * longer matches currentTurnId → stale.
 *
 * Exact sequence from production:
 *   151885: turn_started(osxghp8wux0k)
 *   151886: user_message
 *   151888: interrupt
 *   151889: user_message
 *   151891: turn_started(rxafl22aulgl)       ← new turn before old completes
 *   151892: turn_completed(osxghp8wux0k)     ← STALE
 */
describe('interrupt stale turn race (mnb0dvn5 reproduction)', () => {
  const mkUserMessage = (id: string, text: string) => ({
    type: 'user_message' as const,
    messageId: id,
    forkId: null,
    timestamp: Date.now(),
    content: [{ type: 'text' as const, text }],
    attachments: [] as never[],
    mode: 'text' as const,
    synthetic: false,
    taskMode: false,
  })

  it.live('interrupt + new user message must not produce stale turn_completed', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Turn A starts
      yield* h.send(mkTurnStarted({ turnId: 'turn-A', chainId: 'chain-1' }))

      // User sends a message while turn A is in-flight
      yield* h.send(mkUserMessage('msg-1', 'first message'))

      // User interrupts
      yield* h.send({ type: 'interrupt', forkId: null })

      // Second user message arrives
      yield* h.send(mkUserMessage('msg-2', 'second message'))

      // Complete interrupted turn A first
      yield* h.send(mkTurnCompletedSuccess({ turnId: 'turn-A', chainId: 'chain-1' }))

      // Then turn B starts for queued work
      yield* h.send(mkTurnStarted({ turnId: 'turn-B', chainId: 'chain-2' }))

      // Send a stale/duplicate completion for turn A while B is active
      yield* h.send(mkTurnCompletedSuccess({ turnId: 'turn-A', chainId: 'chain-1' }))

      // Projection must keep active turn B unchanged
      const turn = yield* getTurn(h, null)
      if (turn._tag !== 'active') {
        throw new Error(`expected active turn, got ${turn._tag}`)
      }
      expect(turn.turnId).toBe('turn-B')
    }).pipe(Effect.provide(TestHarnessLive({ workers: { turnController: false } })))
  )

  it.live('worker integration: interrupt + user_message does not race turn IDs', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Send initial user message to start a turn via TurnController
      yield* h.send(mkUserMessage('msg-w1', 'start working'))

      // Wait for the turn to start
      const firstTurn = yield* h.wait.event('turn_started', (e) => e.forkId === null)

      // Wait for it to complete (mock model will auto-complete)
      yield* h.wait.event('turn_completed', (e) => e.forkId === null && e.turnId === firstTurn.turnId)

      // Now simulate the race: send message, interrupt, send another message rapidly
      yield* h.send(mkUserMessage('msg-w2', 'do something'))

      const secondTurn = yield* h.wait.event('turn_started', (e) => e.forkId === null && e.turnId !== firstTurn.turnId)

      // Interrupt while turn is in-flight
      yield* h.send({ type: 'interrupt', forkId: null })

      // Send new message immediately after interrupt
      yield* h.send(mkUserMessage('msg-w3', 'new instruction'))

      // Wait for the interrupted turn to complete
      yield* h.wait.event('turn_completed', (e) => e.forkId === null && e.turnId === secondTurn.turnId)

      // Check: every turn_completed must match the active turn at that point
      const events = eventsForFork(h, null)
      assertNoTurnIdMismatch(events)
    }).pipe(Effect.provide(TestHarnessLive()))
  )
})
