import { beforeEach, describe, expect, it, vi } from 'vitest'

const addLlmClient = vi.fn()
const setPrimary = vi.fn()

vi.mock('@magnitudedev/llm-core', () => ({
  ClientRegistry: class {
    addLlmClient(...args: any[]) {
      addLlmClient(...args)
    }
    setPrimary(...args: any[]) {
      setPrimary(...args)
    }
  },
}))

describe('buildClientRegistry codex oauth parity', () => {
  beforeEach(() => {
    addLlmClient.mockReset()
    setPrimary.mockReset()
  })

  it('sets store=false for openai oauth via openai-responses', async () => {
    const { buildClientRegistry } = await import('../client-registry-builder')

    const registry = buildClientRegistry(
      'openai',
      'gpt-5.4',
      { type: 'oauth', accessToken: 'token', refreshToken: 'refresh', expiresAt: Date.now() + 60_000 },
      undefined,
      undefined,
    )

    expect(registry).toBeTruthy()
    expect(addLlmClient).toHaveBeenCalledTimes(1)

    const [, provider, options] = addLlmClient.mock.calls[0]
    expect(provider).toBe('openai-responses')
    expect(options.base_url).toBe('https://chatgpt.com/backend-api/codex')
    expect(options.store).toBe(false)
  })

  it('passes through instructions but filters rememberedModelIds for openai-responses', async () => {
    const { buildClientRegistry } = await import('../client-registry-builder')

    const registry = buildClientRegistry(
      'openai',
      'gpt-5.4',
      { type: 'api', key: 'sk-test' },
      {
        baseUrl: 'https://api.openai.com/v1',
        instructions: 'SYSTEM PROMPT',
        rememberedModelIds: ['gpt-5.4'],
      } as any,
      undefined,
    )

    expect(registry).toBeTruthy()
    expect(addLlmClient).toHaveBeenCalledTimes(1)

    const [, provider, options] = addLlmClient.mock.calls[0]
    expect(provider).toBe('openai-responses')
    expect(options.instructions).toBe('SYSTEM PROMPT')
    expect(options.rememberedModelIds).toBeUndefined()
  })
})
