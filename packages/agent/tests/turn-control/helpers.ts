import { Effect } from 'effect'
import { expect } from 'vitest'
import type { AppEvent } from '../../src/events'
import { createId } from '../../src/util/id'
import { WorkingStateProjection } from '../../src/projections/working-state'
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
  responseParts: [],
  toolCalls: [],
  observedResults: [],
  result: { success: true, turnDecision: 'yield' },
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
    result: { success: false, error: 'failed', cancelled: false },
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

export const getWorking = (h: Harness, forkId: string | null = null) =>
  h.projectionFork(WorkingStateProjection.Tag, forkId)

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
      current = null
    }
  }
}

export const assertWorkingStateAligned = (h: Harness, forkId: string | null = null) =>
  Effect.gen(function* () {
    const working = yield* getWorking(h, forkId)
    const turn = yield* getTurn(h, forkId)
    expect(working.currentTurnId ?? null).toBe(turn.activeTurn?.turnId ?? null)
  })
