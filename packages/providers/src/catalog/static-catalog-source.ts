import { Effect } from 'effect'
import { getStaticProviderModels } from '../registry'
import type { CatalogSource } from './contracts'

export const staticCatalogSource: CatalogSource = {
  id: 'static',
  priority: 0,
  supports: () => true,
  refresh: (provider) => Effect.succeed([...getStaticProviderModels(provider.id)]),
}
