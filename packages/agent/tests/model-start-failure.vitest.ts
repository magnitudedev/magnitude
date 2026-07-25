import { describe, expect, it } from 'vitest'
import { StreamStartOperationalFailure } from '@magnitudedev/ai'
import {
  agentModelStartRetryability,
  finalizeAgentModelStartFailure,
} from '../src/errors'
import { ModelRequestPreparationFailed } from '../src/model/model-request-preparation'

describe('agent model start failures', () => {
  it('classifies preparation failure as model-not-ready without automatic retry', () => {
    const failure = new ModelRequestPreparationFailed({
      code: 'low_memory',
      message: 'Model stopped · Low memory - close memory-intensive apps and try again',
      retryable: true,
    })

    const decision = finalizeAgentModelStartFailure({
      failure,
      retryCount: 0,
      maxRetries: 3,
    })

    expect(decision.outcome).toEqual({
      _tag: 'ModelNotReady',
      failure: {
        code: 'low_memory',
        message: failure.message,
        retryable: true,
      },
      requestId: null,
    })
    expect(decision.retry).toEqual({ _tag: 'none' })
    expect(agentModelStartRetryability(failure)._tag).toBe('UpstreamNotRetryable')
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
