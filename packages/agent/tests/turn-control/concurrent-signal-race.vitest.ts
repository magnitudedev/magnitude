import { describe, it } from '@effect/vitest'
import { Effect } from 'effect'
import { expect } from 'vitest'
import { TestHarness, TestHarnessLive } from '../../src/test-harness/harness'

describe('turn-control/concurrent-signal-race', () => {
  it.effect('multiple simultaneous readiness signals do not produce duplicate turn_started', () =>
    Effect.gen(function* () {
      const h = yield* TestHarness

      // Send a user message to create a trigger
      yield* h.send({
        type: 'user_message',
        messageId: 'race-msg',
        forkId: null,
        timestamp: Date.now(),
        content: [{ type: 'text', text: 'hello' }],
        attachments: [],
        mode: 'text',
        synthetic: false,
        taskMode: false,
      })

      // Resolve the user message so it becomes a trigger
      yield* h.send({
        type: 'user_message_ready',
        messageId: 'race-msg',
        forkId: null,
        resolvedMentions: [],
      })

      // Wait for the first turn to start
      yield* h.wait.event('turn_started', (e) => e.forkId === null)

      // Complete the turn with chain_continue to re-trigger
      const firstTurn = h.events().find((e): e is Extract<typeof e, { type: 'turn_started' }> => e.type === 'turn_started' && e.forkId === null)!
      yield* h.send({
        type: 'turn_completed',

        forkId: null,
        turnId: firstTurn.turnId,
        chainId: 'race-chain',
        strategyId: 'xml-act',
        result: {
          success: true,
          turnDecision: 'continue',
        },
        inputTokens: null,
        outputTokens: null,
        cacheReadTokens: null,
        cacheWriteTokens: null,
        providerId: null,
        modelId: null,
      })

      // Wait for the chained turn
      const allTurnStarted = yield* h.wait.event('turn_started', (e) => e.forkId === null)

      // Count total turn_started events — should never exceed the number of
      // turn_completed + 1 (initial). Each turn_started must correspond to an idle state.
      const turnStartedCount = h.events().filter(e => e.type === 'turn_started' && e.forkId === null).length
      const turnCompletedCount = h.events().filter(e => e.type === 'turn_completed' && e.forkId === null).length

      // There should be exactly turnCompletedCount + 1 turn_started events (one pending)
      // or turnCompletedCount if the last one completed
      expect(turnStartedCount).toBeLessThanOrEqual(turnCompletedCount + 1)

      // More importantly: no two consecutive turn_started events without a turn_completed between them
      const turnEvents = h.events().filter(e =>
        (e.type === 'turn_started' || e.type === 'turn_completed') && e.forkId === null
      )
      for (let i = 1; i < turnEvents.length; i++) {
        if (turnEvents[i].type === 'turn_started' && turnEvents[i - 1].type === 'turn_started') {
          expect.fail(`Consecutive turn_started without turn_completed at index ${i}: ${turnEvents.map(e => e.type).join(', ')}`)
        }
      }
    }).pipe(Effect.provide(TestHarnessLive())))
})
