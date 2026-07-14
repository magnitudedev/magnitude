import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import { FetchHttpClient } from '@effect/platform'

import { ProviderClient } from '@magnitudedev/sdk'
import {
  resolveSlotModel,
  type SlotId,
} from '@magnitudedev/roles'
import {
  type ModelProfile,
  type ProviderModel,
} from '@magnitudedev/ai'
import {
  computeContextLimits,
  DEFAULT_CONTEXT_LIMIT_POLICY,
  type ResolvedContextLimitPolicy,
  type MagnitudeStorageShape,
  type ModelConfig,
} from '@magnitudedev/storage'
import {
  ROLE_TO_SLOT,
  SLOT_IDS,
  DEFAULT_REASONING_EFFORT,
  resolveReasoningEffort,
  type RoleId,
} from '@magnitudedev/roles'

import { OUTPUT_TOKEN_RESERVE } from '../constants'

export interface SlotConfig {
  readonly slotId: SlotId
  readonly providerId: string
  readonly providerModelId: string
  readonly profile: ModelProfile
  readonly hardCap: number
  readonly softCap: number
  readonly reasoningEffort: string
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

export function buildConfigState<T extends ProviderModel & { readonly slots?: readonly SlotId[] }>(
  catalogModels: readonly T[] | null,
  userConfig: ModelConfig | null,
  policy: ResolvedContextLimitPolicy,
): ConfigState {
  const bySlot = {} as Record<SlotId, SlotConfig>
  for (const slotId of SLOT_IDS) {
    const userSlotConfig = userConfig?.slots?.[slotId]
    const resolved = resolveSlotModel(catalogModels, userSlotConfig, slotId)
    if (resolved === null) {
      throw new NoModelForSlotError(slotId)
    }

    const reasoningEffort = resolveReasoningEffort(
      { reasoningEfforts: resolved.reasoningEfforts },
      userSlotConfig?.reasoningEffort,
      DEFAULT_REASONING_EFFORT[slotId],
    )

    const hardCap = resolved.profile.contextWindow - OUTPUT_TOKEN_RESERVE
    const { softCap } = computeContextLimits(hardCap, policy)

    bySlot[slotId] = {
      slotId,
      providerId: resolved.providerId,
      providerModelId: resolved.providerModelId,
      profile: resolved.profile,
      hardCap,
      softCap,
      reasoningEffort,
      isUserOverride: resolved.isUserOverride,
      isFallback: resolved.isFallback,
    }
  }
  return { bySlot, catalogLoaded: catalogModels !== null }
}

export const ConfigAmbient = Ambient.define<ConfigState, never>({
  name: 'Config',
  initial: Effect.succeed({
    bySlot: {} as Record<SlotId, SlotConfig>,
    catalogLoaded: false,
  }),
})

export function publishConfigFromCatalog(storage: MagnitudeStorageShape) {
  return Effect.gen(function* () {
    const client = yield* ProviderClient
    const ambientService = yield* AmbientServiceTag

    const models = yield* client.catalog.list.pipe(
      Effect.provide(FetchHttpClient.layer),
      Effect.catchAll((err) =>
        Effect.logWarning(`Failed to fetch model catalog: ${err}`)
          .pipe(Effect.as(null))
      ),
    )

    const policy = yield* storage.config.getContextLimitPolicy()
    const modelConfig = yield* storage.config.getModelConfig()
    const newState = Effect.sync(() => buildConfigState(models, modelConfig, policy))

    yield* ambientService.update(
      ConfigAmbient,
      yield* newState,
    )
  })
}
