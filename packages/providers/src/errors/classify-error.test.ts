import { describe, expect, test } from 'bun:test'
import {
  classifyHttpError,
} from './classify-error'
import {
  AuthFailed,
  ContextLimitExceeded,
  RateLimited,
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
})
