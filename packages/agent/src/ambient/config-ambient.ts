import { Ambient, AmbientServiceTag } from '@magnitudedev/event-core'
import { Effect } from 'effect'
import { ProviderState } from '@magnitudedev/providers'
import type { ProviderStateShape } from '@magnitudedev/providers/src/runtime/contracts'

import { MAGNITUDE_SLOTS, type MagnitudeSlot } from '../model-slots'

export interface SlotConfig {
  readonly providerId: string | null
  readonly modelId: string | null
  readonly hardCap: number
  readonly softCap: number
}

export interface ConfigState {
  readonly bySlot: Readonly<Record<MagnitudeSlot, SlotConfig>>
}

export function getSlotConfig(state: ConfigState, slot: MagnitudeSlot): SlotConfig {
  return state.bySlot[slot]
}

export function buildConfigState(opts: {
  providerState: ProviderStateShape<MagnitudeSlot>
}) {
  const { providerState } = opts

  return Effect.gen(function* () {
    const entries = yield* Effect.forEach(
      MAGNITUDE_SLOTS,
      (slot) =>
        Effect.gen(function* () {
          const peek = yield* providerState.peek(slot)
          const { hardCap, softCap } = yield* providerState.contextLimits(slot)

          const config: SlotConfig = {
            providerId: peek?.model.providerId ?? null,
            modelId: peek?.model.id ?? null,
            hardCap,
            softCap,
          }

          return [slot, config] as const
        }),
    )

    const bySlot = {} as Record<MagnitudeSlot, SlotConfig>
    for (const [slot, config] of entries) {
      bySlot[slot] = config
    }

    return {
      bySlot,
    }
  })
}

export const ConfigAmbient = Ambient.define<ConfigState, ProviderStateShape<MagnitudeSlot>>({
  name: 'Config',
  initial: Effect.gen(function* () {
    const providerState = yield* ProviderState
    return yield* buildConfigState({
      providerState: providerState as ProviderStateShape<MagnitudeSlot>,
    })
  }),
})

export function publishConfig(opts: {
  providerState: ProviderStateShape<MagnitudeSlot>
}) {
  return Effect.gen(function* () {
    const ambientService = yield* AmbientServiceTag
    const state = yield* buildConfigState(opts)
    yield* ambientService.update(ConfigAmbient, state)
  })
}

export const publishConfigFromProviders = Effect.gen(function* () {
  const providerState = yield* ProviderState
  yield* publishConfig({
    providerState: providerState as ProviderStateShape<MagnitudeSlot>,
  })
})
