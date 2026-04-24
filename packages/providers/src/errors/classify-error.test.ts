import { describe, expect, test } from 'bun:test'
import {
  classifyHttpError,
} from './classify-error'
import {
  AuthFailed,
  ContextLimitExceeded,
  RateLimited,
  UsageLimitExceeded,
  SubscriptionRequired,
} from './model-error'

describe('classifyHttpError', () => {
  test('maps 401 and 403 to AuthFailed', () => {
    expect(classifyHttpError(401, 'unauthorized')).toBeInstanceOf(AuthFailed)
    expect(classifyHttpError(403, 'forbidden')).toBeInstanceOf(AuthFailed)
  })

  test('maps auth-signal text to AuthFailed even on non-401/403 status', () => {
    const e1 = classifyHttpError(400, 'missing_scope: api.responses.write')
    const e2 = classifyHttpError(500, 'invalid_token')
    const e3 = classifyHttpError(429, 'authentication failed: token expired')
    expect(e1).toBeInstanceOf(AuthFailed)
    expect(e2).toBeInstanceOf(AuthFailed)
    expect(e3).toBeInstanceOf(AuthFailed)
  })

  test('keeps context-limit precedence over auth-like words', () => {
    const e = classifyHttpError(401, 'maximum context length reached, unauthorized')
    expect(e).toBeInstanceOf(ContextLimitExceeded)
  })

  test('keeps 429 as RateLimited when no auth signal', () => {
    const e = classifyHttpError(429, 'rate limit exceeded, retry-after 2s')
    expect(e).toBeInstanceOf(RateLimited)
  })

  test('extracts user-facing message from JSON body for UsageLimitExceeded', () => {
    const rawText = `BamlClientError: BamlClientHttpError: Request failed with status code: 429 Too Many Requests, {"error":{"message":"Weekly usage limit ($50) exceeded for subscription plan.","type":"rate_limit_error","code":"usage_limit_exceeded_weekly","param":null,"details":{"category":"usage_limit_exceeded"}}}`
    const e = classifyHttpError(429, rawText)
    expect(e).toBeInstanceOf(UsageLimitExceeded)
    expect(e.message).toBe('Weekly usage limit ($50) exceeded for subscription plan.')
    expect(e.code).toBe('usage_limit_exceeded_weekly')
  })

  test('falls back to raw text when JSON has error.code but no error.message for UsageLimitExceeded', () => {
    const rawText = `{"error":{"type":"rate_limit_error","code":"usage_limit_exceeded_weekly"}}`
    const e = classifyHttpError(429, rawText)
    expect(e).toBeInstanceOf(UsageLimitExceeded)
    expect(e.message).toBe(rawText)
  })

  test('extracts user-facing message from JSON body for SubscriptionRequired', () => {
    const rawText = `{"error":{"message":"Your subscription has ended. Please upgrade.","code":"subscription_required"}}`
    const e = classifyHttpError(402, rawText)
    expect(e).toBeInstanceOf(SubscriptionRequired)
    expect(e.message).toBe('Your subscription has ended. Please upgrade.')
  })

  test('falls back to raw text when JSON has no error.message for SubscriptionRequired', () => {
    const rawText = `{"some":"other data"}`
    const e = classifyHttpError(402, rawText)
    expect(e).toBeInstanceOf(SubscriptionRequired)
    expect(e.message).toBe(rawText)
  })
})
