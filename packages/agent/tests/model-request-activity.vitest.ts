import { describe, expect, it } from 'vitest'
import {
  initialModelRequestActivityState,
  reduceModelRequestActivity,
} from '../src/display/model-request-activity'

describe('model request activity', () => {
  it('preserves request identity when admission hands off to the native request', () => {
    const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }
    const preparingState = reduceModelRequestActivity(
      initialModelRequestActivityState(),
      {
        turn,
        progress: { phase: 'preparing', requestId: null },
      },
    )
    const queuedState = reduceModelRequestActivity(preparingState, {
      turn,
      progress: { phase: 'queued', requestId: 'request-1' },
    })
    const active = {
      preparing: preparingState.requests.get('root'),
      queued: queuedState.requests.get('root'),
    }

    expect(active.preparing?.requestId).toBeNull()
    expect(active.queued).toMatchObject({
      requestId: 'request-1',
      phase: 'queued',
    })
  })

  it('keeps newer activity when an older request clears late', () => {
    const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }
    const first = reduceModelRequestActivity(initialModelRequestActivityState(), {
      turn,
      progress: { phase: 'queued', requestId: 'request-1' },
    })
    const second = reduceModelRequestActivity(first, {
      turn,
      progress: { phase: 'queued', requestId: 'request-2' },
    })
    const active = reduceModelRequestActivity(second, {
      turn,
      progress: { phase: 'cleared', requestId: 'request-1' },
    }).requests

    expect(active.get('root')).toMatchObject({ requestId: 'request-2' })
  })

  it('removes activity when generation starts', () => {
    const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }
    const prefill = reduceModelRequestActivity(
      initialModelRequestActivityState(),
      {
        turn,
        progress: {
          phase: 'prefill',
          requestId: 'request-1',
          completedTokens: 9_400,
          totalTokens: 14_300,
          cachedTokens: 0,
        },
      },
    )
    const result = reduceModelRequestActivity(prefill, {
      turn,
      progress: {
        phase: 'generating',
        requestId: 'request-1',
      },
    })

    expect(result.requests.has('root')).toBe(false)
  })
})
