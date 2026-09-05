import { describe, expect, it } from 'vitest'
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from '@magnitudedev/providers/client'
import {
  sameConfigStateValue,
  type ConfigState,
  type SlotConfig,
} from '../src/ambient/config-ambient'

const configState = (
  overrides: Partial<SlotConfig> = {},
): ConfigState => ({
  bySlot: {
    primary: {
      _tag: 'Ready',
      config: {
        slotId: 'primary',
        providerId: ProviderIdSchema.make('provider'),
        providerModelId: ProviderModelIdSchema.make('model'),
        modelDisplayName: 'Model',
        profile: { contextWindow: 32_768, maxOutputTokens: 4_096 },
        vision: false,
        hardCap: 28_672,
        softCap: 25_804,
        reasoningEffort: ReasoningEffortSchema.make('none'),
        isUserOverride: true,
        isFallback: false,
        ...overrides,
      },
    },
    secondary: { _tag: 'Unassigned', slotId: 'secondary' },
  },
})

describe('agent configuration value equivalence', () => {
  it('detects catalog-derived configuration changes', () => {
    const current = configState()

    expect(sameConfigStateValue(current, configState())).toBe(true)
    expect(sameConfigStateValue(current, configState({
      profile: { contextWindow: 65_536, maxOutputTokens: 8_192 },
    }))).toBe(false)
    expect(sameConfigStateValue(current, configState({ vision: true }))).toBe(false)
    expect(sameConfigStateValue(current, configState({ softCap: 20_000 }))).toBe(false)
  })
})
