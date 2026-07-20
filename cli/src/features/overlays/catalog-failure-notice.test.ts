import { describe, expect, it } from 'vitest'
import {
  ProviderCatalogUnavailable,
  ProviderIdSchema,
} from '@magnitudedev/sdk'
import { getCatalogFailureNotice } from './catalog-failure-notice'

const failure = (providerId: string, message: string) => new ProviderCatalogUnavailable({
  providerId: ProviderIdSchema.make(providerId),
  message,
})

describe('getCatalogFailureNotice', () => {
  it('does not treat missing optional cloud authentication as an error', () => {
    expect(getCatalogFailureNotice([
      failure('magnitude', 'Magnitude authentication is not configured'),
    ], true)).toBeNull()
  })

  it('still reports other failures when optional cloud authentication is missing', () => {
    expect(getCatalogFailureNotice([
      failure('magnitude', 'Magnitude authentication is not configured'),
      failure('local', 'Local inference unavailable'),
    ], false)).toEqual({
      message: 'Some model providers are currently unavailable.',
      tone: 'warning',
    })
  })

  it('uses concise generic copy for a partially available catalog', () => {
    expect(getCatalogFailureNotice([
      failure('local', 'Local inference unavailable'),
    ], false)).toEqual({
      message: 'Some model providers are currently unavailable.',
      tone: 'warning',
    })
  })

  it('uses concise generic copy when the entire catalog is unavailable', () => {
    expect(getCatalogFailureNotice([
      failure('local', 'Local inference unavailable'),
    ], true)).toEqual({
      message: 'No model providers are currently available.',
      tone: 'error',
    })
  })
})
