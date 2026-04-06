import { describe, expect, it } from 'vitest'
import { Model } from '../model/model'
import { expectedDriverForSelection } from './live-integration-harness'

describe('expectedDriverForSelection', () => {
  it('always returns baml driver for codex oauth selections', () => {
    const model = new Model({
      id: 'gpt-5.4',
      providerId: 'openai',
      name: 'gpt-5.4',
      contextWindow: 200000,
      maxOutputTokens: null,
      costs: null,
    })

    const driver = expectedDriverForSelection(model, { type: 'oauth', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 100000 })

    expect(driver).toBe('baml')
  })
})
