import { Option } from 'effect'
import { describe, expect, it } from 'vitest'
import {
  ModelInstanceIdSchema,
  ModelServingConfigurationIdSchema,
  ModelSlotConfiguredLocal,
  ModelSlotConfiguredRemote,
  ModelSlotUnassigned,
  ModelVariantLabelSchema,
  PRIMARY_SLOT_ID,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  SECONDARY_SLOT_ID,
  type ModelSlotsState,
  type ProviderModelCatalogEntry,
  type SlotSelection,
} from '@magnitudedev/sdk'
import { buildConfigStateFromSlots } from '../src/ambient/config-ambient'

describe('agent model configuration boundary', () => {
  const instanceId = ModelInstanceIdSchema.make("test-instance")
  const configurationId = ModelServingConfigurationIdSchema.make("configuration")

  it('preserves pending slot availability', () => {
    const reasoningEffort = ReasoningEffortSchema.make('none')
    const remoteSelection: SlotSelection = {
      providerId: ProviderIdSchema.make('magnitude'),
      providerModelId: ProviderModelIdSchema.make('remote:model'),
      reasoningEffort,
    }
    const policy = { softCapRatio: 0.9, softCapMaxTokens: 200_000 }
    const slots: ModelSlotsState['slots'] = {
      primary: new ModelSlotConfiguredRemote({
        slotId: PRIMARY_SLOT_ID,
        selection: remoteSelection,
        descriptor: {
          providerId: remoteSelection.providerId,
          providerModelId: remoteSelection.providerModelId,
          displayName: 'Remote model',
          variantLabel: Option.none(),
        },
        availability: { _tag: 'Pending' },
        actions: [],
      }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    }

    expect(buildConfigStateFromSlots(
      [],
      slots,
      policy,
    ).bySlot.primary).toEqual({ _tag: 'Pending', slotId: 'primary' })
  })

  it.each([
    ['not loaded', Option.none()],
    ['loading', Option.some({
      id: instanceId,
      configurationId,
      lifecycle: {
        _tag: 'Loading' as const,
        stage: 'loading' as const,
        progress: Option.some(0.42),
        plannedAllocation: Option.none(),
      },
    })],
    ['stopping', Option.some({
      id: instanceId,
      configurationId,
      lifecycle: {
        _tag: 'Stopping' as const,
        reason: 'user_stop' as const,
        allocation: { _tag: 'Planned' as const, allocation: Option.none() },
      },
    })],
    ['failed', Option.some({
      id: instanceId,
      configurationId,
      lifecycle: {
        _tag: 'Failed' as const,
        failure: { code: 'load_failed', message: 'failed', retryable: true },
      },
    })],
  ] as const)('keeps a selected %s local model callable through the provider boundary', (_state, instance) => {
    const providerId = ProviderIdSchema.make('local')
    const providerModelId = ProviderModelIdSchema.make('local:model')
    const reasoningEffort = ReasoningEffortSchema.make('none')
    const catalog: readonly ProviderModelCatalogEntry[] = [{
      providerId,
      providerModelId,
      modelFamilyId: Option.none(),
      displayName: 'Local model',
      variantLabel: Option.some(ModelVariantLabelSchema.make('Q4 QAT')),
      supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
      contextWindow: 8_192,
      maxOutputTokens: 1_024,
      memory: Option.none(),
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: true,
        reasoning: {
          supported: true,
          efforts: [reasoningEffort],
          defaultEffort: Option.some(reasoningEffort),
        },
      },
      availability: { _tag: 'Available' },
      pricing: Option.none(),
    }]
    const slots: ModelSlotsState['slots'] = {
      primary: new ModelSlotConfiguredLocal({
        slotId: PRIMARY_SLOT_ID,
        selection: { providerId, providerModelId, reasoningEffort },
        descriptor: {
          providerId,
          providerModelId,
          displayName: 'Local model',
          variantLabel: Option.some(ModelVariantLabelSchema.make('Q4 QAT')),
        },
        availability: { _tag: 'Available' },
        instance,
        actions: [],
      }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    }

    const state = buildConfigStateFromSlots(catalog, slots, {
      softCapRatio: 0.9,
      softCapMaxTokens: 200_000,
    })

    expect(state.bySlot.primary).toMatchObject({
      _tag: 'Ready',
      config: {
        providerId,
        providerModelId,
        modelDisplayName: 'Local model (Q4 QAT)',
      },
    })
  })

  it('distinguishes an unassigned slot from a selected unavailable provider', () => {
    const providerId = ProviderIdSchema.make('custom:openrouter')
    const providerModelId = ProviderModelIdSchema.make('z-ai/glm-5.2')
    const reasoningEffort = ReasoningEffortSchema.make('high')
    const catalog: readonly ProviderModelCatalogEntry[] = [{
      providerId,
      providerModelId,
      modelFamilyId: Option.none(),
      displayName: 'GLM 5.2',
      variantLabel: Option.none(),
      supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
      contextWindow: 1048576,
      maxOutputTokens: 128000,
      memory: Option.none(),
      capabilities: {
        vision: false,
        tools: true,
        structuredOutput: false,
        reasoning: {
          supported: true,
          efforts: [reasoningEffort],
          defaultEffort: Option.some(reasoningEffort),
        },
      },
      availability: { _tag: 'Available' },
      pricing: Option.none(),
    }]
    const selection: SlotSelection = { providerId, providerModelId, reasoningEffort }
    const slots: ModelSlotsState['slots'] = {
      primary: new ModelSlotConfiguredRemote({
        slotId: PRIMARY_SLOT_ID,
        selection,
        descriptor: {
          providerId,
          providerModelId,
          displayName: 'GLM 5.2',
          variantLabel: Option.none(),
        },
        availability: {
          _tag: 'Unavailable',
          failure: {
            code: 'provider_not_configured',
            message: 'Set OPENROUTER_API_KEY and restart Magnitude',
            retryable: false,
          },
        },
        actions: [],
      }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    }

    const state = buildConfigStateFromSlots(catalog, slots, {
      softCapRatio: 0.9,
      softCapMaxTokens: 200_000,
    })

    expect(state.bySlot.primary).toEqual({
      _tag: 'Unavailable',
      slotId: 'primary',
      failure: {
        code: 'provider_not_configured',
        message: 'Set OPENROUTER_API_KEY and restart Magnitude',
        retryable: false,
      },
    })
    expect(state.bySlot.secondary).toEqual({
      _tag: 'Unassigned',
      slotId: 'secondary',
    })
  })
})
