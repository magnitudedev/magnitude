import { describe, it, expect } from 'vitest'
import type { MagnitudeSlot } from '@magnitudedev/agent'

type TestSlot = 'lead' | 'worker'
import { getProvider, getStaticProviderModels } from '../registry'

const rest = (model: string): Record<TestSlot, string> => ({
  lead: model,
  worker: model,
})

const tiered = (lead: string, worker: string): Record<TestSlot, string> => ({
  lead,
  worker,
})

const MODEL_DEFAULTS: Record<string, Record<TestSlot, string>> = {
  'anthropic': tiered('claude-opus-4-7', 'claude-sonnet-4-6'),
  'openai': rest('gpt-5.5'),
  'openrouter': tiered('z-ai/glm-5.1', 'moonshotai/kimi-k2.6'),
  'vercel': tiered('zai/glm-5.1', 'moonshotai/kimi-k2.6'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': rest('glm-5.1'),
  'zai-coding-plan': rest('glm-5.1'),
  'moonshotai': rest('kimi-k2.6'),
  'kimi-for-coding': rest('k2p6'),
  'fireworks-ai': rest('accounts/fireworks/models/kimi-k2p6'),
  'magnitude': tiered('glm-5.1', 'kimi-k2.6'),
}

const MODEL_OAUTH_DEFAULTS: Record<string, Record<TestSlot, string>> = {
  'openai': rest('gpt-5.5'),
}

describe('MODEL_DEFAULTS consistency with static registry', () => {
  it('registers Fireworks AI with curated OpenAI-compatible static config', () => {
    const provider = getProvider('fireworks-ai')
    expect(provider).toBeDefined()
    expect(provider?.name).toBe('Fireworks AI')
    expect(provider?.resolveProtocol(null).bamlProvider).toBe('openai-generic')
    expect(provider?.defaultBaseUrl).toBe('https://api.fireworks.ai/inference/v1')
    expect(provider?.authMethods).toEqual([
      { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
    ])
    expect(provider?.providerFamily).toBe('cloud')
    expect(provider?.inventoryMode).toBe('dynamic')

    const staticModels = getStaticProviderModels('fireworks-ai')
    expect(staticModels.map((model) => model.id)).toEqual([
      'accounts/fireworks/models/kimi-k2p6',
      'accounts/fireworks/models/glm-5p1',
    ])
    expect(staticModels.some((model) => model.id === 'accounts/fireworks/models/kimi-k2p6')).toBe(true)
  })

  it('registers Magnitude with static OpenAI-compatible API-key config', () => {
    const provider = getProvider('magnitude')
    expect(provider).toBeDefined()
    expect(provider?.name).toBe('Magnitude')
    expect(provider?.resolveProtocol(null).bamlProvider).toBe('openai-generic')
    expect(provider?.defaultBaseUrl).toBe('https://app.magnitude.dev/api/v1')
    expect(provider?.authMethods).toEqual([
      { type: 'api-key', label: 'API key', envKeys: ['MAGNITUDE_API_KEY'] },
    ])
    expect(provider?.providerFamily).toBe('cloud')
    expect(provider?.inventoryMode).toBe('static')

    const staticModels = getStaticProviderModels('magnitude')
    expect(staticModels.map((model) => model.id)).toEqual([
      'glm-4.7',
      'glm-5',
      'glm-5.1',
      'kimi-k2.5',
      'kimi-k2.6',
      'minimax-m2.5',
      'minimax-m2.7',
    ])
  })

  for (const [providerId, slotMap] of Object.entries(MODEL_DEFAULTS)) {
    for (const [slot, modelId] of Object.entries(slotMap)) {
      it(`${providerId}/${slot}: "${modelId}" exists in static registry`, () => {
        const staticModels = getStaticProviderModels(providerId)
        expect(staticModels.some(m => m.id === modelId)).toBe(true)
      })
    }
  }

  for (const [providerId, slotMap] of Object.entries(MODEL_OAUTH_DEFAULTS)) {
    for (const [slot, modelId] of Object.entries(slotMap)) {
      it(`oauth:${providerId}/${slot}: "${modelId}" exists in static registry`, () => {
        const staticModels = getStaticProviderModels(providerId)
        expect(staticModels.some(m => m.id === modelId)).toBe(true)
      })
    }
  }
})
