import { describe, expect, it } from 'vitest'
import { AuthFailed, ProviderDisconnected } from '@magnitudedev/providers'
import {
  classifyRetryability,
  resolveFailureMessage,
} from '../src/workers/cortex-auth'

describe('cortex auth reconnect messaging', () => {
  it('resolveFailureMessage maps AuthFailed to reconnect guidance', () => {
    const message = resolveFailureMessage(
      new AuthFailed({ message: 'missing_scope: api.responses.write' }),
    )
    expect(message).toBe('Your provider session expired or became invalid. Please reconnect in /settings.')
  })

  it('resolveFailureMessage preserves ProviderDisconnected reconnect message', () => {
    const message = resolveFailureMessage(
      new ProviderDisconnected({
        providerId: 'openai',
        providerName: 'OpenAI',
        message: 'OpenAI session expired or became invalid. Please reconnect in /settings.',
      }),
    )
    expect(message).toBe('OpenAI session expired or became invalid. Please reconnect in /settings.')
  })

  it('classifyRetryability marks AuthFailed cause as auth', () => {
    const reason = classifyRetryability(
      new AuthFailed({ message: 'token expired' }),
    )
    expect(reason).toBe('auth')
  })
})
