import { isBrowserCompatible } from '@magnitudedev/providers'
import type { ProviderDefinition } from '@magnitudedev/providers'

export function pickLocalBrowserModel(
  providerId: string,
  models: ProviderDefinition['models'],
): ProviderDefinition['models'][number] | null {
  return models.find((model) => isBrowserCompatible(providerId, model.id)) ?? models[0] ?? null
}
