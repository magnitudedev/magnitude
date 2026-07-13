import { describe, expect, it } from 'vitest'
import { StreamOperationalFailure, acceptedHttpResponse, snapshotModelAttemptFailure, type ProviderCall } from '@magnitudedev/ai'

describe('snapshotModelAttemptFailure', () => {
  it('captures stream operational failure details without response headers', () => {
    const call: ProviderCall = {
      provider: 'test',
      model: 'model',
      method: 'POST',
      url: 'https://example.test',
    }

    const failure = new StreamOperationalFailure({
      call,
      response: acceptedHttpResponse(200, new Headers()),
      reason: {
        _tag: 'BodyReadFailure',
        readError: {
          _tag: 'EffectResponseBodyError',
          effectReason: 'Decode',
          cause: { _tag: 'ErrorCause', name: 'TypeError', message: 'terminated' },
        },
      },
      progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 2 },
    })

    const snapshot = snapshotModelAttemptFailure(failure)

    expect(snapshot.phase).toBe('stream')
    expect(snapshot.tag).toBe('StreamOperationalFailure')
    expect(snapshot.detailTag).toBe('BodyReadFailure')
    expect(snapshot.responseStatus).toBe(200)
    expect(snapshot.progress).toEqual({ dataPayloadsDecoded: 1, modelEventsEmitted: 2 })
    expect(snapshot.retryable).toBe(true)
  })
})
