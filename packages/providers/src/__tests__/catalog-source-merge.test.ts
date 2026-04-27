import { describe, expect, it } from 'vitest'

import { mergeProviderModels } from '../catalog/merge'
import type { ProviderModel } from '../model/model'

function model(id: string, overrides: Partial<ProviderModel> = {}): ProviderModel {
  return {
    id,
    providerId: 'test',
    providerName: 'Test',
    modelId: null,
    name: id,
    contextWindow: 1000,
    maxContextTokens: null,
    maxOutputTokens: null,
    supportsToolCalls: false,
    supportsReasoning: false,
    supportsVision: false,
    costs: { inputPerM: 0, outputPerM: 0, cacheReadPerM: null, cacheWritePerM: null },
    releaseDate: '2025-01-01T00:00:00.000Z',
    discovery: { primarySource: 'static' },
    ...overrides,
  }
}

describe('catalog source merge order', () => {
  it('preserves static-only ids while overlaying later sources', () => {
    const merged = mergeProviderModels(
      [],
      [
        model('accounts/fireworks/models/kimi-k2p6'),
        model('accounts/fireworks/models/glm-5p1', { name: 'Static GLM' }),
      ],
      [
        model('accounts/fireworks/models/glm-5p1', {
          name: 'Live GLM',
          supportsToolCalls: true,
          discovery: { primarySource: 'models.dev' },
        }),
      ],
    )

    expect(merged.some((entry) => entry.id === 'accounts/fireworks/models/kimi-k2p6')).toBe(true)
    expect(merged.find((entry) => entry.id === 'accounts/fireworks/models/glm-5p1')?.name).toBe('Live GLM')
  })

  it('later sources only override defined fields', () => {
    const merged = mergeProviderModels(
      [],
      [model('x', { supportsVision: true })],
      [model('x', { supportsVision: undefined, supportsToolCalls: true, discovery: { primarySource: 'models.dev' } })],
    )

    expect(merged[0]?.supportsToolCalls).toBe(true)
    expect(merged[0]?.supportsVision).toBe(true)
  })
})
