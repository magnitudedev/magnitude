import { Effect, Option, Schema } from 'effect'
import { ProviderIdSchema, ProviderModelIdSchema, ReasoningEffortSchema } from '@magnitudedev/providers/client'
import { describe, expect, it } from 'vitest'
import {
  PRIMARY_SLOT_ID,
  ProviderModelIdentitySchema,
  SlotSelectionSchema,
} from '@magnitudedev/acn-protocol'
import { ModelStateSchema } from '@magnitudedev/storage'

import { makeModelSelection } from './model-selection'
import { makeTestModelState } from './model-state.test-support'

const identity = (providerId: string, providerModelId: string) =>
  ProviderModelIdentitySchema.make({
    providerId: ProviderIdSchema.make(providerId),
    providerModelId: ProviderModelIdSchema.make(providerModelId),
  })

const selection = (providerId: string, providerModelId: string) =>
  SlotSelectionSchema.make({
    providerId: ProviderIdSchema.make(providerId),
    providerModelId: ProviderModelIdSchema.make(providerModelId),
    reasoningEffort: ReasoningEffortSchema.make('none'),
  })

const emptyState = Schema.decodeUnknownSync(ModelStateSchema)({
})

describe('ModelSelection', () => {
  it('stores complete slot identity and provider-qualified bounded recency', async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const state = yield* makeTestModelState(emptyState)
      const service = makeModelSelection(state)
      yield* service.updateSlot(PRIMARY_SLOT_ID, Option.some(selection('local', 'model-a')))
      yield* service.recordUse(PRIMARY_SLOT_ID, identity('remote', 'model-a'))
      return yield* service.get
    }))

    expect(result.slots.primary).toEqual(Option.some(selection('local', 'model-a')))
    expect(result.recentModels.primary).toEqual([
      identity('remote', 'model-a'),
      identity('local', 'model-a'),
    ])
  })

  it('stores favorites by complete provider identity', async () => {
    const result = await Effect.runPromise(Effect.gen(function* () {
      const state = yield* makeTestModelState(emptyState)
      const service = makeModelSelection(state)
      yield* service.setFavorite(identity('local', 'shared'), true)
      yield* service.setFavorite(identity('remote', 'shared'), true)
      yield* service.setFavorite(identity('local', 'shared'), false)
      return yield* service.get
    }))
    expect(result.favorites).toEqual([identity('remote', 'shared')])
  })
})
