import { describe, expect, it } from 'vitest'
import { CanonicalAccumulatorReducer, projectCanonical } from '../reducers'
import type { HarnessEvent, TurnOutcome } from '../../events'

describe('canonical turn reducer', () => {
  it('preserves StreamFailed terminal details in canonical outcome', () => {
    const terminal = {
      _tag: 'StreamFailed',
      cause: {
        _tag: 'ProviderStreamError',
        message: 'socket closed',
      },
      usage: {
        _tag: 'UsageNotReported',
        reason: 'stream_failed',
      },
    }

    const outcome: TurnOutcome = {
      _tag: 'StreamFailed',
      requestId: 'req-1',
      message: 'stream failed',
      terminal: terminal as any,
    }

    const event: HarnessEvent = {
      _tag: 'TurnEnd',
      outcome,
      usage: null,
    }

    const state = CanonicalAccumulatorReducer.step(CanonicalAccumulatorReducer.initial, event)
    expect(projectCanonical(state).outcome).toEqual(outcome)
  })
})
