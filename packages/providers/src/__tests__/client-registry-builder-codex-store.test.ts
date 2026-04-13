import { afterEach, describe, expect, it, vi } from 'vitest'
import { __testOnly_buildProviderOptions } from '../client-registry-builder'

describe('client-registry codex openai-responses options', () => {
  it('openai oauth codex enforces store:false and passes instructions/headers', () => {
    const options = __testOnly_buildProviderOptions(
      'openai',
      'gpt-5.3-codex',
      { type: 'oauth', oauthMethod: 'oauth-browser', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 60_000, accountId: 'acct' },
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
      { type: 'oauth', oauthMethod: 'oauth-device', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 60_000 },
      { instructions: 'SYSTEM' },
    )

    expect(options).toBeDefined()
    expect(options?.base_url).toBe('https://api.githubcopilot.com')
    expect(options?.instructions).toBe('SYSTEM')
    expect(options?.store).toBeUndefined()
  })
})

describe('client-registry openai-generic Fireworks options', () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('builds Fireworks options from explicit api auth', () => {
    const options = __testOnly_buildProviderOptions(
      'fireworks-ai',
      'accounts/fireworks/routers/kimi-k2p5-turbo',
      { type: 'api', key: 'test-key' },
      undefined,
      ['STOP'],
    )

    expect(options).toBeDefined()
    expect(options).toEqual(expect.objectContaining({
      model: 'accounts/fireworks/routers/kimi-k2p5-turbo',
      api_key: 'test-key',
      base_url: 'https://api.fireworks.ai/inference/v1',
      max_tokens: 131072,
      stop: ['STOP'],
      stream_options: { include_usage: true },
    }))
  })

  it('uses FIREWORKS_API_KEY and honors baseUrl override through generic path', () => {
    vi.stubEnv('FIREWORKS_API_KEY', 'env-fireworks-key')

    const options = __testOnly_buildProviderOptions(
      'fireworks-ai',
      'accounts/fireworks/models/glm-5p1',
      null,
      { baseUrl: 'https://example.fireworks.test/v1' },
    )

    expect(options).toBeDefined()
    expect(options).toEqual(expect.objectContaining({
      model: 'accounts/fireworks/models/glm-5p1',
      api_key: 'env-fireworks-key',
      base_url: 'https://example.fireworks.test/v1',
      max_tokens: 131072,
      stream_options: { include_usage: true },
    }))
  })
})
