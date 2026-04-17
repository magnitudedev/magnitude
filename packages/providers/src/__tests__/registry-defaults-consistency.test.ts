import { describe, it, expect } from 'vitest'
import type { MagnitudeSlot } from '@magnitudedev/agent'

type TestSlot = 'lead' | 'worker' | 'browser'
import { getProvider, getStaticProviderModels } from '../registry'

const rest = (model: string): Record<TestSlot, string> => ({
  lead: model,
  worker: model,
  browser: model,
})

const tiered = (lead: string, sub: string, browser: string): Record<TestSlot, string> => ({
  lead,
  worker: sub,
  browser,
})

const MODEL_DEFAULTS: Record<string, Record<TestSlot, string>> = {
  'anthropic': tiered('claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5'),
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
  'openrouter': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'vercel': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': rest('glm-4.7'),
  'zai-coding-plan': rest('glm-4.7'),
  'moonshotai': rest('kimi-k2.5'),
  'kimi-for-coding': rest('k2p5'),
  'fireworks-ai': rest('accounts/fireworks/routers/kimi-k2p5-turbo'),
  'magnitude': rest('glm-5.1'),
}

const MODEL_OAUTH_DEFAULTS: Record<string, Record<TestSlot, string>> = {
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
}

describe('MODEL_DEFAULTS consistency with static registry', () => {
  it('registers Fireworks AI with curated OpenAI-compatible static config', () => {
    const provider = getProvider('fireworks-ai')
    expect(provider).toBeDefined()
    expect(provider?.name).toBe('Fireworks AI')
    expect(provider?.bamlProvider).toBe('openai-generic')
    expect(provider?.defaultBaseUrl).toBe('https://api.fireworks.ai/inference/v1')
    expect(provider?.authMethods).toEqual([
      { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
    ])
    expect(provider?.providerFamily).toBe('cloud')
    expect(provider?.inventoryMode).toBe('dynamic')

    const staticModels = getStaticProviderModels('fireworks-ai')
    expect(staticModels.map((model) => model.id)).toEqual([
      'accounts/fireworks/routers/kimi-k2p5-turbo',
      'accounts/fireworks/models/glm-5p1',
    ])
    expect(staticModels.some((model) => model.id === 'accounts/fireworks/routers/kimi-k2p5-turbo')).toBe(true)
  })

  it('registers Magnitude with static OpenAI-compatible API-key config', () => {
    const provider = getProvider('magnitude')
    expect(provider).toBeDefined()
    expect(provider?.name).toBe('Magnitude')
    expect(provider?.bamlProvider).toBe('openai-generic')
    expect(provider?.defaultBaseUrl).toBe('https://app.magnitude.dev/api/v1')
    expect(provider?.authMethods).toEqual([
      { type: 'api-key', label: 'API key', envKeys: ['MAGNITUDE_API_KEY'] },
    ])
    expect(provider?.providerFamily).toBe('cloud')
    expect(provider?.inventoryMode).toBe('static')

    const staticModels = getStaticProviderModels('magnitude')
    expect(staticModels.map((model) => model.id)).toEqual([
      'qwen3.6-plus',
      'glm-4.7',
      'glm-5',
      'glm-5.1',
      'kimi-k2.5',
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
