import { describe, expect, it } from 'vitest'

import { mergeProviderModels } from '../catalog/merge'
import type { ModelDefinition } from '../types'

function model(id: string, overrides: Partial<ModelDefinition> = {}): ModelDefinition {
  return {
    id,
    name: id,
    contextWindow: 1000,
    supportsToolCalls: false,
    supportsReasoning: false,
    cost: { input: 0, output: 0 },
    family: 'test',
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
      [model('x', { description: 'static', supportsVision: true })],
      [model('x', { description: undefined, supportsToolCalls: true, discovery: { primarySource: 'models.dev' } })],
    )

    expect(merged[0]?.description).toBe('static')
    expect(merged[0]?.supportsToolCalls).toBe(true)
  })
})
