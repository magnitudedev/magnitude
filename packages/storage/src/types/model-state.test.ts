import { Option, Schema } from 'effect'
import { describe, expect, it } from 'vitest'

import { ModelStateSchema } from './model-state'

const decode = Schema.decodeUnknownSync(ModelStateSchema)

describe('ModelStateSchema', () => {
  it('materializes the complete canonical state', () => {
    expect(decode({})).toEqual({
      slots: { primary: Option.none(), secondary: Option.none() },
      recentModels: { primary: [], secondary: [] },
      favorites: [],
    })
  })

  it('discards obsolete derived configuration fields', () => {
    expect(decode({
      configurations: [{ obsolete: true }],
      configurationRecoveryCompleted: true,
    })).toEqual({
      slots: { primary: Option.none(), secondary: Option.none() },
      recentModels: { primary: [], secondary: [] },
      favorites: [],
    })
  })

  it('deduplicates provider-qualified preferences and bounds recency', () => {
    const identities = Array.from({ length: 40 }, (_, index) => ({
      providerId: index === 1 ? 'other' : 'local',
      providerModelId: index === 1 ? 'same' : `model-${index}`,
    }))
    const state = decode({
      recentModels: {
        primary: [identities[0], identities[0], ...identities.slice(1)],
        secondary: [],
      },
      favorites: [identities[0], identities[0], identities[1]],
    })
    expect(state.recentModels.primary).toHaveLength(32)
    expect(state.favorites).toEqual([identities[0], identities[1]])
  })
})
