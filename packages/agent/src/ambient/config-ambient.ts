import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect, Option } from 'effect'

import {
  ModelCatalogLifecycle,
  ModelSlotsLifecycle,
  type ModelCatalog,
  type ModelSlots,
  type ModelSummary,
  type SlotStates,
} from '@magnitudedev/sdk'
import { type SlotId } from '@magnitudedev/roles'
import { type ReasoningEffort } from '@magnitudedev/ai'
import {
  computeContextLimits,
  DEFAULT_CONTEXT_LIMIT_POLICY,
  type ResolvedContextLimitPolicy,
  type MagnitudeStorageShape,
} from '@magnitudedev/storage'
import { ROLE_TO_SLOT, SLOT_IDS, type RoleId } from '@magnitudedev/roles'

import { OUTPUT_TOKEN_RESERVE } from '../constants'

export interface SlotConfig {
  readonly slotId: SlotId
  readonly providerId: string
  readonly providerModelId: string
  readonly profile: { readonly contextWindow: number; readonly maxOutputTokens: number }
  readonly vision: boolean | undefined
  readonly hardCap: number
  readonly softCap: number
  readonly reasoningEffort: ReasoningEffort
  readonly isUserOverride: boolean
  readonly isFallback: boolean
}

export interface ConfigState {
  readonly bySlot: Readonly<Record<SlotId, SlotConfig>>
  readonly catalogLoaded: boolean
}

export function getSlotConfig(state: ConfigState, slotId: SlotId): SlotConfig {
  return state.bySlot[slotId]
}

export function getSlotConfigForRole(state: ConfigState, roleId: RoleId): SlotConfig {
  const slotId = ROLE_TO_SLOT[roleId]
  return state.bySlot[slotId]
}

export class NoModelForSlotError extends Error {
  constructor(
    public readonly slotId: SlotId,
  ) {
    super(`No model available for slot ${slotId}. Check your API key and model configuration.`)
    this.name = 'NoModelForSlotError'
  }
}

export const ConfigAmbient = Ambient.define<ConfigState, never>({
  name: 'Config',
  initial: Effect.succeed({
    bySlot: {} as Record<SlotId, SlotConfig>,
    catalogLoaded: false,
  }),
})

export function buildConfigStateFromSlots(
  catalogModels: readonly ModelSummary[],
  slots: SlotStates,
  policy: ResolvedContextLimitPolicy,
): ConfigState {
  const bySlot = {} as Record<SlotId, SlotConfig>
  for (const slotId of SLOT_IDS) {
    const slot = slots[slotId]
    if (slot._tag !== 'Ready') throw new NoModelForSlotError(slotId)
    const hardCap = slot.contextWindow - OUTPUT_TOKEN_RESERVE
    const { softCap } = computeContextLimits(hardCap, policy)
    const selectedModel = catalogModels.find((model) => model.providerId === slot.selection.providerId
      && model.providerModelId === slot.selection.providerModelId)
    const visionProperty = selectedModel?.properties.vision
    const vision = visionProperty?._tag === 'Cached' || visionProperty?._tag === 'Resolved' || visionProperty?._tag === 'Refreshing'
      ? visionProperty.value
      : undefined
    bySlot[slotId] = {
      slotId,
      providerId: slot.selection.providerId,
      providerModelId: slot.selection.providerModelId,
      profile: { contextWindow: slot.contextWindow, maxOutputTokens: slot.maxOutputTokens },
      vision,
      hardCap,
      softCap,
      reasoningEffort: slot.selection.reasoningEffort,
      isUserOverride: slot.source === 'user',
      isFallback: false,
    }
  }
  return { bySlot, catalogLoaded: true }
}

export function publishConfigFromModelResources(
  storage: MagnitudeStorageShape,
  modelCatalog: Effect.Effect<ModelCatalog>,
  modelSlots: Effect.Effect<ModelSlots>,
) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    const catalog = yield* modelCatalog
    const slotSnapshot = yield* modelSlots
    const models = ModelCatalogLifecycle.match(catalog.state, {
      loading: () => [] as readonly ModelSummary[],
      ready: ({ models }) => models,
      refreshing: ({ models }) => models,
      degraded: ({ models }) => models,
      unavailable: () => [] as readonly ModelSummary[],
    })
    const slots = Option.match(ModelSlotsLifecycle.match(slotSnapshot.state, {
      loading: () => Option.none<SlotStates>(),
      ready: ({ slots }) => Option.some(slots),
      refreshing: ({ slots }) => Option.some(slots),
      degraded: ({ slots }) => Option.some(slots),
      unavailable: ({ slots }) => Option.some(slots),
    }), {
      onNone: () => { throw new NoModelForSlotError('primary') },
      onSome: (value) => value,
    })
    const policy = yield* storage.config.getContextLimitPolicy()
    const newState = Effect.sync(() => buildConfigStateFromSlots(models, slots, policy))

    yield* ambientService.update(
      ConfigAmbient,
      yield* newState,
    )
  })
}
