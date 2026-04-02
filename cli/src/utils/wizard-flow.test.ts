import { describe, expect, test } from 'bun:test'
import { computeWizardTotalSteps, resolveLocalWizardSlotDefaults, resolveWizardBackStep } from './wizard-flow'

describe('computeWizardTotalSteps', () => {
  test('local provider branch without browser uses 3 steps', () => {
    expect(computeWizardTotalSteps(false, true)).toBe(3)
  })

  test('non-local branch without browser uses 2 steps', () => {
    expect(computeWizardTotalSteps(false, false)).toBe(2)
  })

  test('local provider branch with browser uses 4 steps', () => {
    expect(computeWizardTotalSteps(true, true)).toBe(4)
  })
})

describe('resolveWizardBackStep', () => {
  test('models goes back to provider-endpoint on local branch', () => {
    expect(resolveWizardBackStep('models', true)).toBe('provider-endpoint')
  })

  test('models goes back to provider on non-local branch', () => {
    expect(resolveWizardBackStep('models', false)).toBe('provider')
  })

  test('provider-endpoint goes back to provider', () => {
    expect(resolveWizardBackStep('provider-endpoint', true)).toBe('provider')
  })
})

describe('resolveLocalWizardSlotDefaults', () => {
  const slots = ['lead', 'explorer', 'planner', 'builder', 'reviewer', 'debugger', 'browser'] as const
  const emptyModels = {
    lead: null, explorer: null, planner: null, builder: null, reviewer: null, debugger: null, browser: null,
  }

  test('assigns one selected model to all seven roles when defaults are applied', () => {
    const result = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'lmstudio',
      existingSlotModels: emptyModels,
      discoveredModelIds: ['qwen2.5-coder'],
      rememberedModelIds: [],
      applyWizardDefaults: true,
    })

    for (const slot of slots) {
      expect(result[slot]).toEqual({ providerId: 'lmstudio', modelId: 'qwen2.5-coder' })
    }
  })

  test('uses precedence: saved provider model > discovered > manual > unset', () => {
    const savedWins = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'ollama',
      existingSlotModels: {
        ...emptyModels,
        planner: { providerId: 'ollama', modelId: 'saved-ollama-model' },
      },
      discoveredModelIds: ['discovered-model'],
      rememberedModelIds: ['manual-model'],
      applyWizardDefaults: true,
    })
    expect(savedWins.lead).toEqual({ providerId: 'ollama', modelId: 'saved-ollama-model' })

    const discoveredWins = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'ollama',
      existingSlotModels: emptyModels,
      discoveredModelIds: ['discovered-model'],
      rememberedModelIds: ['manual-model'],
      applyWizardDefaults: true,
    })
    expect(discoveredWins.lead).toEqual({ providerId: 'ollama', modelId: 'discovered-model' })

    const manualWins = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'ollama',
      existingSlotModels: emptyModels,
      discoveredModelIds: [],
      rememberedModelIds: ['manual-model'],
      applyWizardDefaults: true,
    })
    expect(manualWins.lead).toEqual({ providerId: 'ollama', modelId: 'manual-model' })

    const unsetWhenEmpty = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'ollama',
      existingSlotModels: emptyModels,
      discoveredModelIds: [],
      rememberedModelIds: [],
      applyWizardDefaults: true,
    })
    for (const slot of slots) {
      expect(unsetWhenEmpty[slot]).toBeNull()
    }
  })

  test('does not overwrite existing saved slot assignments when defaults are not explicitly applied', () => {
    const existing = {
      lead: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      explorer: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      planner: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      builder: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      reviewer: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      debugger: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
      browser: { providerId: 'openrouter', modelId: 'anthropic/claude-sonnet-4.6' },
    }

    const result = resolveLocalWizardSlotDefaults({
      slots,
      providerId: 'lmstudio',
      existingSlotModels: existing,
      discoveredModelIds: ['qwen2.5-coder'],
      rememberedModelIds: ['manual-model'],
      applyWizardDefaults: false,
    })

    expect(result).toEqual(existing)
  })
})
