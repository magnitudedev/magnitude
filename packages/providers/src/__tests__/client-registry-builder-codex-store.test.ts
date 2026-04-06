import { describe, expect, it } from 'vitest'
import { __testOnly_buildProviderOptions } from '../client-registry-builder'

describe('client-registry codex openai-responses options', () => {
  it('openai oauth codex enforces store:false and passes instructions/headers', () => {
    const options = __testOnly_buildProviderOptions(
      'openai',
      'gpt-5.3-codex',
      { type: 'oauth', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 60_000, accountId: 'acct' },
      {
        instructions: 'SYSTEM',
        store: false,
        headers: { 'X-Test': '1' },
        rememberedModelIds: ['secret-internal'],
      },
    )

    expect(options).toBeDefined()
    expect(options?.store).toBe(false)
    expect(options?.instructions).toBe('SYSTEM')
    expect(options?.headers?.['X-Test']).toBe('1')
    expect(options?.rememberedModelIds).toBeUndefined()
  })

  it('copilot codex uses openai-responses-safe options without forcing store', () => {
    const options = __testOnly_buildProviderOptions(
      'github-copilot',
      'gpt-5-codex',
      { type: 'oauth', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 60_000 },
      { instructions: 'SYSTEM' },
    )

    expect(options).toBeDefined()
    expect(options?.base_url).toBe('https://api.githubcopilot.com')
    expect(options?.instructions).toBe('SYSTEM')
    expect(options?.store).toBeUndefined()
  })
})
