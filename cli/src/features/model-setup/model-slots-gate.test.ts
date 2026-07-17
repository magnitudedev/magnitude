import { describe, expect, it } from 'vitest'
import {
  ModelSlotConfigurationUnavailable,
  ProviderIdSchema,
  ProviderCatalogUnavailable,
} from '@magnitudedev/sdk'
import { blockingModelSlotsFailure } from './model-slots-gate'

describe('blockingModelSlotsFailure', () => {
  it('does not block when an optional provider is not authenticated', () => {
    expect(blockingModelSlotsFailure({
      failures: [new ProviderCatalogUnavailable({
        providerId: ProviderIdSchema.make('magnitude'),
        message: 'Magnitude authentication is not configured',
      })],
    })).toBeNull()
  })

  it('blocks when the saved model configuration cannot be read', () => {
    expect(blockingModelSlotsFailure({
      failures: [new ModelSlotConfigurationUnavailable({
        message: 'Could not read config.json',
      })],
    })).toBe('Could not read config.json')
  })

  it('reports only blocking failures from a mixed failure set', () => {
    expect(blockingModelSlotsFailure({
      failures: [
        new ProviderCatalogUnavailable({
          providerId: ProviderIdSchema.make('magnitude'),
          message: 'Magnitude authentication is not configured',
        }),
        new ModelSlotConfigurationUnavailable({
          message: 'Could not read config.json',
        }),
      ],
    })).toBe('Could not read config.json')
  })
})
