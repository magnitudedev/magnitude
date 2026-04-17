import { MODELS_DEV_TTL_MS } from '@magnitudedev/storage'
import { Effect } from 'effect'

import type { CatalogSource } from './contracts'
import { fetchModelsDevData, normalizeModelsDevProvider } from './models-dev-source'
import { resolveCachedSource } from './source-cache'

export const modelsDevCatalogSource: CatalogSource = {
  id: 'models.dev',
  priority: 100,
  supports: (provider) => provider.providerFamily !== 'local' && provider.id !== 'magnitude',
  refresh: (provider) =>
    resolveCachedSource(
      'models-dev',
      MODELS_DEV_TTL_MS,
      fetchModelsDevData,
    ).pipe(
      Effect.map((data) => (data ? normalizeModelsDevProvider(provider.id, data) : [])),
    ),
}
