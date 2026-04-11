import { describe, it, expect } from 'vitest'
import type { MagnitudeSlot } from '@magnitudedev/agent'
import { getProvider, getStaticProviderModels } from '../registry'

const rest = (model: string): Record<MagnitudeSlot, string> => ({
  lead: model,
  explorer: model,
  planner: model,
  builder: model,
  reviewer: model,
  debugger: model,
  browser: model,
})

const tiered = (lead: string, sub: string, browser: string): Record<MagnitudeSlot, string> => ({
  lead,
  explorer: sub,
  planner: sub,
  builder: sub,
  reviewer: sub,
  debugger: sub,
  browser,
})

/** Duplicated from cli/src/utils/model-preferences.ts for validation */
const MODEL_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'anthropic': tiered('claude-opus-4-6', 'claude-sonnet-4-6', 'claude-haiku-4-5'),
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
  'github-copilot': tiered('claude-opus-4.6', 'claude-sonnet-4.6', 'claude-haiku-4.5'),
  'openrouter': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'vercel': tiered('anthropic/claude-opus-4.6', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-haiku-4.5'),
  'amazon-bedrock': tiered('us.anthropic.claude-opus-4-6-v1', 'us.anthropic.claude-sonnet-4-6-v1', 'us.anthropic.claude-haiku-4-5-v1'),
  'google-vertex-anthropic': tiered('claude-opus-4-6@default', 'claude-sonnet-4-6@default', 'claude-haiku-4-5@default'),
  'google': tiered('gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3-flash-preview'),
  'google-vertex': tiered('gemini-3.1-pro-preview', 'gemini-3-flash-preview', 'gemini-3-flash-preview'),
  'cerebras': rest('gpt-oss-120b'),
  'minimax': rest('MiniMax-M2.7'),
  'zai': rest('glm-4.7'),
  'zai-coding-plan': rest('glm-4.7'),
  'moonshotai': rest('kimi-k2.5'),
  'kimi-for-coding': rest('k2p5'),
  'fireworks': rest('accounts/fireworks/routers/kimi-k2p5-turbo'),
}

/** Duplicated from cli/src/utils/model-preferences.ts for validation */
const MODEL_OAUTH_DEFAULTS: Record<string, Record<MagnitudeSlot, string>> = {
  'openai': tiered('gpt-5.4', 'gpt-5.3-codex', 'gpt-5.3-codex'),
}

describe('MODEL_DEFAULTS consistency with static registry', () => {
  it('registers Fireworks AI with curated OpenAI-compatible static config', () => {
    const provider = getProvider('fireworks')
    expect(provider).toBeDefined()
    expect(provider?.name).toBe('Fireworks AI')
    expect(provider?.bamlProvider).toBe('openai-generic')
    expect(provider?.defaultBaseUrl).toBe('https://api.fireworks.ai/inference/v1')
    expect(provider?.authMethods).toEqual([
      { type: 'api-key', label: 'API key', envKeys: ['FIREWORKS_API_KEY'] },
    ])

    const staticModels = getStaticProviderModels('fireworks')
    expect(staticModels.map((model) => model.id)).toEqual([
      'accounts/fireworks/routers/kimi-k2p5-turbo',
      'accounts/fireworks/models/glm-5p1',
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
