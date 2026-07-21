import { describe, expect, it } from 'vitest'
import type { ModelAttemptFailureSnapshot } from '@magnitudedev/ai'
import { present } from '../src/errors/present'

const failure = (providerMessage?: string): ModelAttemptFailureSnapshot => ({
  phase: 'stream',
  tag: 'StreamProviderError',
  detailTag: 'invalid_request',
  message: 'Model stream ended with provider error envelope\nmessage: prompt exceeds context',
  ...(providerMessage ? { providerMessage } : {}),
  call: { provider: 'local', model: 'model-1', method: 'POST', url: 'icn://chat/model-1' },
  responseStatus: 200,
  progress: null,
  retryable: false,
  retryAfterMs: null,
})

describe('error presentation', () => {
  it('shows the exact provider reason for an in-band provider error', () => {
    expect(present({ _tag: 'StreamFailed', failure: failure('prompt exceeds context'), requestId: null }).message)
      .toBe('prompt exceeds context')
  })

  it('falls back to the diagnostic instead of a generic stream label', () => {
    expect(present({ _tag: 'StreamFailed', failure: failure(), requestId: null }).message)
      .toContain('prompt exceeds context')
  })
})
