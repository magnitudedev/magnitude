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
        activity: {
          _tag: 'Preparing',
          preparation: { phase: 'preparing' },
          requestId: null,
        },
      },
    )
    const queuedState = reduceModelRequestActivity(preparingState, {
      turn,
      activity: {
        _tag: 'Preparing',
        preparation: { phase: 'queued' },
        requestId: 'request-1',
      },
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
      activity: { _tag: 'Preparing', preparation: { phase: 'queued' }, requestId: 'request-1' },
    })
    const second = reduceModelRequestActivity(first, {
      turn,
      activity: { _tag: 'Preparing', preparation: { phase: 'queued' }, requestId: 'request-2' },
    })
    const active = reduceModelRequestActivity(second, {
      turn,
      activity: { _tag: 'Ended', requestId: 'request-1' },
    }).requests

    expect(active.get('root')).toMatchObject({ requestId: 'request-2' })
  })

  it('removes activity when generation starts', () => {
    const turn = { turnId: 'turn-1', chainId: 'chain-1', forkId: null }
    const prefill = reduceModelRequestActivity(
      initialModelRequestActivityState(),
      {
        turn,
        activity: {
          _tag: 'Preparing',
          preparation: {
            phase: 'prefill',
            completed_tokens: 9_400,
            total_tokens: 14_300,
            cached_tokens: 0,
          },
          requestId: 'request-1',
        },
      },
    )
    const result = reduceModelRequestActivity(prefill, {
      turn,
      activity: {
        _tag: 'Streaming',
        requestId: 'request-1',
      },
    })

    expect(result.requests.has('root')).toBe(false)
  })
})
