import type { AmbientService } from '@magnitudedev/event-core'
import {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
} from '@magnitudedev/ai'
import { Effect, Ref } from 'effect'
import { describe, expect, it } from 'vitest'
import { ConfigAmbient, type ConfigState } from '../src/ambient/config-ambient'
import { makeModelConfigurationSynchronizer } from '../src/coding-agent'

const config = (revision: number, providerModelId: string): ConfigState => ({
  revision,
  catalogLoaded: true,
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
    secondary: { _tag: 'Unavailable', slotId: 'secondary', reason: 'not_loaded' },
  },
})

describe('resident session model configuration', () => {
  it('synchronizes a preloaded session before use and rejects delayed older revisions', async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const authoritative = yield* Ref.make(config(2, 'selected-model'))
      let resident = config(1, 'stale-preload-model')
      const applied: number[] = []
      const ambient = {
        register: () => Effect.void,
        getValue: () => resident,
        update: (_definition, state: ConfigState) => Effect.sync(() => {
          resident = state
          applied.push(state.revision)
        }),
      } as AmbientService
      const synchronizer = yield* makeModelConfigurationSynchronizer(
        ambient,
        Ref.get(authoritative),
      )

      yield* synchronizer.sync
      yield* synchronizer.apply(config(1, 'late-stream-model'))
      yield* Ref.set(authoritative, config(3, 'newer-model'))
      yield* synchronizer.sync

      return { resident, applied }
    }))

    expect(result.resident.bySlot.primary._tag).toBe('Ready')
    if (result.resident.bySlot.primary._tag !== 'Ready') return
    expect(result.resident.bySlot.primary.config.providerModelId).toBe('newer-model')
    expect(result.applied).toEqual([2, 3])
  })
})
