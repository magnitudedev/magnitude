import { Effect } from 'effect'
import { expect } from 'vitest'
import type { AppEvent } from '../../src/events'
import { createId } from '../../src/util/id'
import { TurnProjection } from '../../src/projections/turn'
import { TestHarness } from '../../src/test-harness/harness'

export type Harness = Effect.Effect.Success<typeof TestHarness>

export const mkTurnStarted = (
  overrides: Partial<Extract<AppEvent, { type: 'turn_started' }>> = {},
): Extract<AppEvent, { type: 'turn_started' }> => ({
  type: 'turn_started',
  forkId: null,
  turnId: createId(),
  chainId: createId(),
  ...overrides,
})

export const mkTurnCompletedSuccess = (
  overrides: Partial<Extract<AppEvent, { type: 'turn_completed' }>> = {},
): Extract<AppEvent, { type: 'turn_completed' }> => ({
  type: 'turn_completed',
  forkId: null,
  turnId: 'turn-1',
  chainId: 'chain-1',
  strategyId: 'xml-act',
  result: { _tag: 'Completed', completion: { decision: 'idle', feedback: [] } },

  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  providerId: null,
  modelId: null,
  ...overrides,
})

export const mkTurnCompletedFailure = (
  overrides: Partial<Extract<AppEvent, { type: 'turn_completed' }>> = {},
): Extract<AppEvent, { type: 'turn_completed' }> => ({
  ...mkTurnCompletedSuccess({
    result: { _tag: 'SystemError', message: 'failed' },
  }),
  ...overrides,
})

export const mkContextLimitHit = (
  forkId: string | null = null,
  error = 'context exceeded',
): Extract<AppEvent, { type: 'context_limit_hit' }> => ({
  type: 'context_limit_hit',
  forkId,
  error,
})

export const getTurn = (h: Harness, forkId: string | null = null) =>
  h.projectionFork(TurnProjection.Tag, forkId)

export const eventsForFork = (h: Harness, forkId: string | null) =>
  h.events().filter((e) => 'forkId' in e && e.forkId === forkId)

export const assertNoTurnIdMismatch = (events: readonly AppEvent[], forkId: string | null = null) => {
  let current: string | null = null
  for (const event of events) {
    if (!('forkId' in event) || event.forkId !== forkId) continue
    if (event.type === 'turn_started') {
      current = event.turnId
      continue
    }
    if (event.type === 'turn_completed') {
      expect(current).toBe(event.turnId)
      current = null
      continue
    }
    if (event.type === 'turn_unexpected_error') {
      // This event can appear for related forks in transcript ordering.
      // Only enforce if it is for the currently tracked turn.
      if (current !== null && event.turnId === current) {
        current = null
      }
      continue
    }
    if (event.type === 'interrupt') {
      continue
    }
  }
}

export const assertTurnStateAligned = (h: Harness, forkId: string | null = null) =>
  Effect.gen(function* () {
    const turn = yield* getTurn(h, forkId)
    expect(turn._tag === 'idle' || turn.turnId.length > 0).toBe(true)
  })
