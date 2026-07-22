import { ProviderIdSchema, type ProviderCatalogFailure } from '@magnitudedev/sdk'

const MAGNITUDE_PROVIDER_ID = ProviderIdSchema.make('magnitude')

export interface CatalogFailureNotice {
  readonly message: string
  readonly tone: 'warning' | 'error'
}

const isMissingCloudAuthentication = (failure: ProviderCatalogFailure): boolean =>
  failure._tag === 'ProviderFailure'
  && failure.providerId === MAGNITUDE_PROVIDER_ID
  && failure.message === 'Magnitude authentication is not configured'

export function getCatalogFailureNotice(
  failures: readonly ProviderCatalogFailure[],
  catalogUnavailable: boolean,
): CatalogFailureNotice | null {
  const actionableFailures = failures.filter((failure) => !isMissingCloudAuthentication(failure))
  if (actionableFailures.length === 0) return null

  return catalogUnavailable
    ? { message: 'No model providers are currently available.', tone: 'error' }
    : { message: 'Some model providers are currently unavailable.', tone: 'warning' }
}
