import { Option, Schema } from 'effect'
import { describe, expect, it } from 'vitest'

import { ModelStateSchema } from './model-state'

const configuration = (id: string, contextLength = 32_768) => ({
  id,
  bundle: {
    _tag: 'Standalone',
    package: {
      id: 'package-a',
      source: { _tag: 'Local', path: '/models/a.gguf' },
      files: [],
      relationships: [],
      properties: {
        format: 'gguf',
        quantization: 'q4',
        quantizationName: 'Q4',
        architecture: 'test',
        maximumContextLength: 131_072,
      },
    },
  },
  profile: { contextLength },
})

const decode = Schema.decodeUnknownSync(ModelStateSchema)

describe('ModelStateSchema', () => {
  it('materializes the complete canonical state', () => {
    expect(decode({})).toEqual({
      configurations: [],
      slots: { primary: Option.none(), secondary: Option.none() },
      recentModels: { primary: [], secondary: [] },
      favorites: [],
      configurationRecoveryCompleted: false,
    })
  })

  it('rejects conflicting configuration identities', () => {
    expect(() => decode({
      configurations: [
        configuration('configuration-a'),
        configuration('configuration-a', 65_536),
      ],
      configurationRecoveryCompleted: true,
    })).toThrow()
  })

  it('rejects multiple retained configurations for one bundle', () => {
    expect(() => decode({
      configurations: [
        configuration('configuration-a'),
        configuration('configuration-b', 65_536),
      ],
      configurationRecoveryCompleted: true,
    })).toThrow()
  })

  it('deduplicates provider-qualified preferences and bounds recency', () => {
    const identities = Array.from({ length: 40 }, (_, index) => ({
      providerId: index === 1 ? 'other' : 'local',
      providerModelId: index === 1 ? 'same' : `model-${index}`,
    }))
    const state = decode({
      configurations: [],
      recentModels: {
        primary: [identities[0], identities[0], ...identities.slice(1)],
        secondary: [],
      },
      favorites: [identities[0], identities[0], identities[1]],
      configurationRecoveryCompleted: true,
    })
    expect(state.recentModels.primary).toHaveLength(32)
    expect(state.favorites).toEqual([identities[0], identities[1]])
  })
})
