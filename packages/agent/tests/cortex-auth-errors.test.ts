import { describe, expect, it } from 'vitest'
import {
  AuthFailed,
  ProviderDisconnected,
  UsageLimitExceeded,
} from '@magnitudedev/providers'
import {
  classifyModelError,
  classifyRetryability,
} from '../src/workers/cortex-auth'

describe('cortex auth reconnect messaging', () => {
  it('classifyModelError maps AuthFailed to ProviderNotReady(AuthFailed)', () => {
    const outcome = classifyModelError(
      new AuthFailed({ message: 'missing_scope: api.responses.write' }),
    )
    expect(outcome).toEqual({
      _tag: 'ProviderNotReady',
      detail: {
        _tag: 'AuthFailed',
        providerId: 'unknown',
        providerName: 'Unknown provider',
      },
    })
  })

  it('classifyModelError preserves ProviderDisconnected provider identity', () => {
    const outcome = classifyModelError(
      new ProviderDisconnected({
        providerId: 'openai',
        providerName: 'OpenAI',
        message: 'OpenAI session expired or became invalid. Please reconnect in /settings.',
      }),
    )
    expect(outcome).toEqual({
      _tag: 'ProviderNotReady',
      detail: {
        _tag: 'ProviderDisconnected',
        providerId: 'openai',
        providerName: 'OpenAI',
      },
    })
  })

  it('classifyRetryability marks AuthFailed cause as auth', () => {
    const reason = classifyRetryability(
      new AuthFailed({ message: 'token expired' }),
    )
    expect(reason).toBe('auth')
  })
})

describe('classifyModelError', () => {
  it('maps UsageLimitExceeded to ProviderNotReady(MagnitudeBilling)', () => {
    const cause = new UsageLimitExceeded({
      message: 'Weekly usage limit ($50) exceeded for subscription plan.',
      code: 'usage_limit_exceeded_weekly',
    })
    const outcome = classifyModelError(cause)
    expect(outcome).toEqual({
      _tag: 'ProviderNotReady',
      detail: {
        _tag: 'MagnitudeBilling',
        reason: { _tag: 'UsageLimitExceeded', message: 'Weekly usage limit ($50) exceeded for subscription plan.' },
      },
    })
  })

  it('uses model error classification directly', () => {
    const cause = new UsageLimitExceeded({
      message: 'Clean user message',
      code: 'usage_limit_exceeded_weekly',
    })
    const outcome = classifyModelError(cause)
    expect(outcome).toEqual({
      _tag: 'ProviderNotReady',
      detail: {
        _tag: 'MagnitudeBilling',
        reason: { _tag: 'UsageLimitExceeded', message: 'Clean user message' },
      },
    })
  })
})
