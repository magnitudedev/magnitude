import { Option } from 'effect'
import { describe, expect, it } from 'vitest'
import {
  LocalModelIdSchema,
  ModelSlotBlocked,
  ModelSlotLoadingLocalModel,
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotUnloadingLocalModel,
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
  it.each([
    ['unloaded', (selection: SlotSelection) => new ModelSlotUnloadedLocalModel({ slotId: PRIMARY_SLOT_ID, selection })],
    ['loading', (selection: SlotSelection) => new ModelSlotLoadingLocalModel({ slotId: PRIMARY_SLOT_ID, selection, percentage: 42 })],
    ['unloading', (selection: SlotSelection) => new ModelSlotUnloadingLocalModel({ slotId: PRIMARY_SLOT_ID, selection })],
    ['blocked', (selection: SlotSelection) => new ModelSlotBlocked({
      slotId: PRIMARY_SLOT_ID,
      selection,
      reason: { _tag: 'LocalModelLoadFailed', error: { code: 'load_failed', message: 'failed', retryable: true } },
    })],
  ] as const)('keeps a selected %s local model callable through the provider boundary', (_state, makeSlot) => {
    const providerId = ProviderIdSchema.make('local')
    const providerModelId = ProviderModelIdSchema.make(`local:${LocalModelIdSchema.make('model')}`)
    const reasoningEffort = ReasoningEffortSchema.make('none')
    const catalog: readonly ProviderModelCatalogEntry[] = [{
      providerId,
      providerModelId,
      modelFamilyId: Option.none(),
      displayName: 'Local model',
      supportedSlots: [PRIMARY_SLOT_ID, SECONDARY_SLOT_ID],
      contextWindow: 8_192,
      maxOutputTokens: 1_024,
      runtimeMemoryBytes: Option.none(),
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
      primary: makeSlot({ providerId, providerModelId, reasoningEffort }),
      secondary: new ModelSlotUnassigned({ slotId: SECONDARY_SLOT_ID }),
    }

    const state = buildConfigStateFromSlots(catalog, slots, {
      softCapRatio: 0.9,
      softCapMaxTokens: 200_000,
    })

    expect(state.bySlot.primary).toMatchObject({
      _tag: 'Ready',
      config: { providerId, providerModelId },
    })
  })
})
