import { describe, expect, it } from 'vitest'
import { StreamProviderError, StreamStartOperationalFailure } from '@magnitudedev/ai'
import {
  agentModelStartRetryability,
  finalizeAgentModelStartFailure,
  finalizeModelAttemptFailure,
} from '../src/errors'

describe('agent model start failures', () => {
  it('presents streamed local admission failures as model-not-ready without automatic retry', () => {
    const failure = new StreamProviderError({
      call: { provider: 'local', model: 'model', method: 'POST', url: 'http://127.0.0.1/inference/v1/chat/completions' },
      response: { status: 200, headers: [], requestId: null },
      providerError: {
        message: 'Not enough memory to load model',
        type: 'model_error',
        code: 'low_memory',
        param: null,
        retryable: true,
      },
      payload: { text: '', encodedBytes: 0, truncated: false },
      progress: { dataPayloadsDecoded: 1, modelEventsEmitted: 0 },
    })

    const decision = finalizeModelAttemptFailure({ failure, retryCount: 0, maxRetries: 3 })

    expect(decision.outcome).toEqual({
      _tag: 'ModelNotReady',
      failure: { code: 'low_memory', message: 'Not enough memory to load model', retryable: true },
      requestId: null,
    })
    expect(decision.retry).toEqual({ _tag: 'none' })
  })

  it('retains existing provider connection retry behavior', () => {
    const failure = new StreamStartOperationalFailure({
      call: {
        provider: 'test',
        model: 'model',
        method: 'POST',
        url: 'https://example.test/chat',
      },
      reason: {
        _tag: 'RequestFailedBeforeResponse',
        cause: {
          _tag: 'ErrorCause',
          name: 'Error',
          message: 'network down',
        },
      },
    })

    const decision = finalizeAgentModelStartFailure({
      failure,
      retryCount: 0,
      maxRetries: 3,
    })

    expect(decision.outcome._tag).toBe('ConnectionFailure')
    expect(decision.retry._tag).toBe('retry')
    expect(agentModelStartRetryability(failure)._tag).toBe('UpstreamRetryable')
  })
})
