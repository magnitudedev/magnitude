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

export const mkTurnOutcomeEventSuccess = (
  overrides: Partial<Extract<AppEvent, { type: 'turn_outcome' }>> = {},
): Extract<AppEvent, { type: 'turn_outcome' }> => ({
  type: 'turn_outcome',
  forkId: null,
  turnId: 'turn-1',
  chainId: 'chain-1',
  strategyId: 'xml-act',
  outcome: { _tag: 'Completed', completion: { yieldTarget: 'user', feedback: [] } },

  inputTokens: null,
  outputTokens: null,
  cacheReadTokens: null,
  cacheWriteTokens: null,
  providerId: null,
  modelId: null,
  ...overrides,
})

export const mkTurnOutcomeEventFailure = (
  overrides: Partial<Extract<AppEvent, { type: 'turn_outcome' }>> = {},
): Extract<AppEvent, { type: 'turn_outcome' }> => ({
  ...mkTurnOutcomeEventSuccess({
    outcome: { _tag: 'UnexpectedError', message: 'failed' },
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
    if (event.type === 'turn_outcome') {
      expect(current).toBe(event.turnId)
      current = null
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
