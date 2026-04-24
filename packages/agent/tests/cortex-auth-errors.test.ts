import { describe, expect, it } from 'vitest'
import { AuthFailed, ProviderDisconnected, UsageLimitExceeded } from '@magnitudedev/providers'
import {
  buildGeneralErrorPayload,
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

describe('buildGeneralErrorPayload', () => {
  it('extracts errorCode from UsageLimitExceeded cause for CTA rendering', () => {
    const cause = new UsageLimitExceeded({
      message: 'Weekly usage limit ($50) exceeded for subscription plan.',
      code: 'usage_limit_exceeded_weekly',
    })
    const { message, errorCode } = buildGeneralErrorPayload(
      cause.message,
      cause,
    )
    expect(errorCode).toBe('usage_limit_exceeded_weekly')
    // Known model errors use resolveFailureMessage — clean message without prefix
    expect(message).toBe('Weekly usage limit ($50) exceeded for subscription plan.')
  })

  it('uses resolveFailureMessage for known model errors, not the passed error text', () => {
    const cause = new UsageLimitExceeded({
      message: 'Clean user message',
      code: 'usage_limit_exceeded_weekly',
    })
    // Pass a noisy raw error, but the cause has a known tag
    const { message, errorCode } = buildGeneralErrorPayload('Some raw BAML crash dump', cause)
    expect(errorCode).toBe('usage_limit_exceeded_weekly')
    expect(message).toBe('Clean user message')
  })

  it('returns undefined errorCode when cause has no code property', () => {
    const cause = new Error('some generic error')
    const { errorCode } = buildGeneralErrorPayload(
      cause.message,
      cause,
    )
    expect(errorCode).toBeUndefined()
  })

  it('returns undefined errorCode when cause is null', () => {
    const { errorCode } = buildGeneralErrorPayload('test error', null)
    expect(errorCode).toBeUndefined()
  })

  it('truncates messages longer than 500 characters', () => {
    const longMessage = 'x'.repeat(1000)
    const { message } = buildGeneralErrorPayload(longMessage, null)
    expect(message.length).toBeLessThanOrEqual(500 + 3) // includes "..."
    expect(message.endsWith('...')).toBe(true)
  })

  it('does not truncate short messages', () => {
    const shortMessage = 'A short error'
    const { message } = buildGeneralErrorPayload(shortMessage, null)
    expect(message.endsWith('...')).toBe(false)
    expect(message).toBe('Unexpected error while executing turn: A short error')
  })

  it('includes errorCode alongside truncated message when both apply (unclassified cause with code)', () => {
    // A cause with a code property but no _tag — falls through to truncation
    const cause = { code: 'usage_limit_exceeded_weekly' }
    const longMessage = 'x'.repeat(1000)
    const { message, errorCode } = buildGeneralErrorPayload(longMessage, cause)
    expect(errorCode).toBe('usage_limit_exceeded_weekly')
    expect(message.length).toBeLessThanOrEqual(503)
    expect(message.endsWith('...')).toBe(true)
    expect(message).toContain('Unexpected error while executing turn:')
  })
})
