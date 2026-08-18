import type { AmbientService } from '@magnitudedev/event-core'
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from '@magnitudedev/ai'
import { Effect, Ref } from 'effect'
import { describe, expect, it } from 'vitest'
import { ConfigAmbient, type ConfigState } from '../src/ambient/config-ambient'
import type { ToolAvailabilityState } from '../src/ambient/tool-availability-ambient'
import {
  makeModelConfigurationSynchronizer,
  makeToolAvailabilitySynchronizer,
} from '../src/coding-agent'

const config = (providerModelId: string): ConfigState => ({
  bySlot: {
    primary: {
      _tag: 'Ready',
      config: {
        slotId: 'primary',
        providerId: ProviderIdSchema.make('local'),
        providerModelId: ProviderModelIdSchema.make(providerModelId),
        modelDisplayName: providerModelId,
        profile: { contextWindow: 100_000, maxOutputTokens: 4_000 },
        vision: false,
        hardCap: 96_000,
        softCap: 80_000,
        reasoningEffort: ReasoningEffortSchema.make('medium'),
        isUserOverride: true,
        isFallback: false,
      },
    },
    secondary: { _tag: 'Pending', slotId: 'secondary' },
  },
})

describe('resident session model configuration', () => {
  it('re-reads the authoritative snapshot and skips semantic duplicates', async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const authoritative = yield* Ref.make(config('selected-model'))
      let resident = config('stale-preload-model')
      const applied: string[] = []
      const ambient = {
        register: () => Effect.void,
        getValue: () => resident,
        update: (_definition, state: ConfigState) => Effect.sync(() => {
          resident = state
          if (state.bySlot.primary._tag === 'Ready') {
            applied.push(state.bySlot.primary.config.providerModelId)
          }
        }),
      } as AmbientService
      const synchronizer = yield* makeModelConfigurationSynchronizer(
        ambient,
        Ref.get(authoritative),
      )

      yield* synchronizer.sync
      yield* synchronizer.sync
      yield* Ref.set(authoritative, config('newer-model'))
      yield* synchronizer.sync

      return { resident, applied }
    }))

    expect(result.resident.bySlot.primary._tag).toBe('Ready')
    if (result.resident.bySlot.primary._tag !== 'Ready') return
    expect(result.resident.bySlot.primary.config.providerModelId).toBe('newer-model')
    expect(result.applied).toEqual(['selected-model', 'newer-model'])
  })
})

describe('resident session tool availability', () => {
  it('re-reads the authoritative snapshot and skips semantic duplicates', async () => {
    const state = (
      source: 'magnitude' | 'exa' | 'unavailable',
    ): ToolAvailabilityState => ({
      webSearch: source === 'unavailable'
        ? { _tag: 'Unavailable' }
        : { _tag: 'Available', source },
    })
    const result = await Effect.runPromise(Effect.gen(function* () {
      const authoritative = yield* Ref.make(state('exa'))
      let resident = state('unavailable')
      const applied: string[] = []
      const ambient = {
        register: () => Effect.void,
        getValue: () => resident,
        update: (_definition, next: ToolAvailabilityState) => Effect.sync(() => {
          resident = next
          applied.push(next.webSearch._tag === 'Available'
            ? next.webSearch.source
            : 'unavailable')
        }),
      } as AmbientService
      const synchronizer = yield* makeToolAvailabilitySynchronizer(
        ambient,
        Ref.get(authoritative),
      )

      yield* synchronizer.sync
      yield* synchronizer.sync
      yield* Ref.set(authoritative, state('magnitude'))
      yield* synchronizer.sync

      return { resident, applied }
    }))

    expect(result.resident).toEqual(state('magnitude'))
    expect(result.applied).toEqual(['exa', 'magnitude'])
  })
})
